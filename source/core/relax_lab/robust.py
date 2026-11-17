# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""General bounded, surface-constrained corner-Jacobian relaxation.

Original implementation motivated by regularized determinant energies, with
aspect-aware quad targets, local L-BFGS proposals and actual-path safeguards.
No third-party solver source is imported. Fixed connectivity is preserved.
"""
import time
import numpy as np
from .solver import PatchSolver, face_values


class RobustSolver(PatchSolver):
    def __init__(self, points, faces, fixed=None):
        faces = [list(f) for f in faces]
        super().__init__(points,faces,fixed)
        # Restore explicit constraints; the legacy cosine solver also locked
        # every vertex of a degenerate face. A usable neighboring frame lets us
        # recover a collapsed edge instead of freezing its entire neighborhood.
        self.fixed = np.zeros(len(points),dtype=bool) if fixed is None else np.array(fixed,dtype=bool)
        self.fixed |= np.array([not ns for ns in self.neighbors])
        for face in faces:
            if len(face) not in (3,4):
                self.fixed[face]=True
        normal_sum = np.zeros_like(self.points)
        initial_values = {}
        for g in self.groups:
            data=face_values(self.points,g["cells"])
            signed=np.sum(np.cross(data[2],data[1])*g["normals"][:,None],axis=2)
            # Both constructor passes use the same initial coordinates/frames.
            initial_values[id(g)] = (data, signed)
            healthy=np.all(signed>np.maximum(data[4].mean(axis=1)[:,None]**2*1e-10,1e-30),axis=1)
            for k in range(g["cells"].shape[1]):
                np.add.at(normal_sum,g["cells"][healthy,k],g["normals"][healthy]*g["area"][healthy,None])
        ids,inverses,normals,corner_groups = [],[],[],[]
        positive_lengths = self.length[self.length>1e-15]
        if not len(positive_lengths):
            raise ValueError("No nonzero edge scale is available")
        minimum_scale=float(np.median(positive_lengths))*1e-6
        for group in self.groups:
            cells=group["cells"]
            data,signed=initial_values.pop(id(group))
            edge_scale=np.maximum(data[4].mean(axis=1),minimum_scale)
            crossed=(signed < -edge_scale[:,None]**2*1e-10).sum(axis=1)>=2
            missing=(np.linalg.norm(group["normals"],axis=1)<.5)|crossed
            inferred_available=np.zeros(len(cells),dtype=bool)
            if missing.any():
                inferred=normal_sum[cells[missing]].sum(axis=1)
                length=np.linalg.norm(inferred,axis=1)
                valid=length>minimum_scale**2*1e-10
                inferred_available[np.flatnonzero(missing)[valid]]=True
                group["normals"][inferred_available]=inferred[valid]/length[valid,None]
            # A crossed face can use a frame supplied by healthy neighbors. An
            # isolated ambiguous winding has no trustworthy repair orientation.
            ambiguous=(crossed & ~inferred_available)|(np.linalg.norm(group["normals"],axis=1)<.5)
            group["valid"]=~ambiguous
            self.fixed[cells[ambiguous].ravel()]=True
            size=cells.shape[1]
            selected=cells[~ambiguous]
            lengths=data[4][~ambiguous]
            if not len(selected):
                continue
            ids.append(np.stack((selected,np.roll(selected,-1,axis=1),np.roll(selected,1,axis=1)),axis=2).reshape(-1,3))
            if size==4:
                uv=np.maximum(np.column_stack(((lengths[:,0]+lengths[:,2])*.5,
                                                (lengths[:,1]+lengths[:,3])*.5)),minimum_scale)
                inverse=np.zeros((len(selected),size,2,2))
                parity=np.arange(size)%2
                inverse[:,:,0,0]=1/uv[:,parity]
                inverse[:,:,1,1]=1/uv[:,1-parity]
            else:
                h=np.maximum(np.exp(np.mean(np.log(np.maximum(lengths,minimum_scale)),axis=1)),minimum_scale)
                inverse=np.broadcast_to(np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]]),
                                        (len(selected),size,2,2))/h[:,None,None,None]
            inverses.append(inverse.reshape(-1,2,2))
            normals.append(np.repeat(group["normals"][~ambiguous],size,axis=0))
            padded=np.pad(selected,((0,0),(0,4-size)),mode="edge")
            corner_groups.append(np.repeat(padded,size,axis=0))
        self.ids=np.ascontiguousarray(np.concatenate(ids) if ids else np.empty((0,3)),dtype=np.int32)
        self.inverse=np.ascontiguousarray(np.concatenate(inverses) if inverses else np.empty((0,2,2)),dtype=np.float64)
        self.corner_normals=np.ascontiguousarray(np.concatenate(normals) if normals else np.empty((0,3)),dtype=np.float64)
        self.corner_owners=np.ascontiguousarray(np.concatenate(corner_groups) if corner_groups else np.empty((0,4)),dtype=np.int32)
        self.vertex_normals=normal_sum/np.maximum(np.linalg.norm(normal_sum,axis=1)[:,None],1e-30)
        self.history=[]
        self._active_corners=None
        self._active_vertices=None
        self.epsilon=None
        self._last_eval=None
        self._metric_cache=None
        self.last=dict(moved=0,backtracks=0,accepted=False,bad_corners=0,state="Ready")

    def drop_last_eval(self):
        """Forget the last energy evaluation (the coordinates were rounded by the mesh storage)."""
        self._last_eval=None

    def reset_stroke(self):
        self.history.clear()
        self.epsilon=None
        self._active_corners=None
        self._active_vertices=None
        self._carry=0.
        self._last_eval=None
        self._metric_cache=None

    def metric(self, points, corner_indices=None, epsilon=1e-6, gradient=False, determinants_only=False):
        if corner_indices is None or len(corner_indices)==len(self.ids):
            ids,inverse,normals=self.ids,self.inverse,self.corner_normals
        else:
            # Only cache the internally generated active set. Coordinates,
            # costs, gradients and determinants are always freshly evaluated.
            cache=self._metric_cache if corner_indices is self._active_corners else None
            if (cache is not None and cache[0] is corner_indices
                    and cache[1] is self.ids and cache[2] is self.inverse
                    and cache[3] is self.corner_normals):
                ids,inverse,normals=cache[4:]
            else:
                ids=np.ascontiguousarray(self.ids[corner_indices])
                inverse=np.ascontiguousarray(self.inverse[corner_indices])
                normals=np.ascontiguousarray(self.corner_normals[corner_indices])
                if corner_indices is self._active_corners:
                    self._metric_cache=(corner_indices,self.ids,self.inverse,self.corner_normals,
                                        ids,inverse,normals)
        if not len(ids):
            return np.empty(0),np.zeros_like(points) if gradient else None,np.empty(0)
        p=points[ids]
        edges=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
        jac=edges@inverse
        det=np.sum(np.cross(jac[:,:,0],jac[:,:,1])*normals,axis=1)
        if determinants_only:
            return None,None,det
        root=np.hypot(det,epsilon)
        chi=.5*(det+root)
        negative=det<0
        chi[negative]=epsilon**2/(2*(root[negative]-det[negative]))
        cost=(np.sum(jac**2,axis=(1,2))+.05*(det**2+1))/chi
        result=None
        if gradient:
            dual=np.stack((np.cross(jac[:,:,1],normals),np.cross(normals,jac[:,:,0])),axis=2)
            derivative=2*jac/chi[:,None,None]+(.1*det/chi-cost/root)[:,None,None]*dual
            edge_gradient=derivative@inverse.transpose(0,2,1)
            local=np.stack((-edge_gradient.sum(axis=2),edge_gradient[:,:,0],edge_gradient[:,:,1]),axis=1)
            result=np.zeros_like(points)
            np.add.at(result,ids.ravel(),(local/len(ids)).reshape(-1,3))
        return cost,result,det

    def _path_safe(self, candidate, indices, start, end):
        middle=self.metric((self.points+candidate)*.5,indices,determinants_only=True)[2]
        a=2*(end+start-2*middle)
        b=end-start-a
        t=np.clip(-b/np.where(np.abs(a)>1e-30,2*a,1.),0,1)
        minimum=np.minimum(start,end)
        minimum=np.where(a>0,np.minimum(minimum,start+b*t+a*t*t),minimum)
        positive=start>1e-10
        return bool(np.all(minimum[positive]>=start[positive]*.01))

    def step(self, weights, dt, strength=.65, pressure=1., method="ROBUST", project=None, passes=1):
        weights=np.clip(np.asarray(weights,dtype=float),0,1).copy()
        weights[self.fixed]=0
        active=weights>0
        if not active.any() or strength<=0 or pressure<=0:
            self.last=dict(moved=0,backtracks=0,accepted=False,bad_corners=0,state="Masked / fixed")
            return self.last
        # Keep all corners of a touched face, including the opposite corner of
        # a quad, even when its three sampled vertices themselves are fixed.
        if self._active_vertices is None or not np.array_equal(active,self._active_vertices):
            self._active_corners=np.flatnonzero(active[self.corner_owners].any(axis=1))
            self._active_vertices=active.copy()
            self.history.clear()
            self._last_eval=None
        indices=self._active_corners
        moved=backtracks=0
        state="Converged / constrained"
        started=time.perf_counter()
        for iteration in range(max(1,passes)):
            if iteration and time.perf_counter()-started>.012:
                break
            cached=self._last_eval
            if cached is not None and cached[0] is self.points and cached[1] is indices:
                determinant=cached[5]
            else:
                cached=None
                determinant=self.metric(self.points,indices,determinants_only=True)[2]
            if not len(determinant):
                break
            epsilon=max(1e-6,.2*abs(min(0.,float(determinant.min()))))
            if self.epsilon is None or abs(np.log(epsilon/self.epsilon))>.25:
                self.history.clear()
                self.epsilon=epsilon
            epsilon=self.epsilon
            if cached is not None and cached[2]==epsilon:
                cost,gradient,determinant=cached[3],cached[4].copy(),cached[5]
            else:
                cost,gradient,determinant=self.metric(self.points,indices,epsilon,True)
                gradient-=(np.sum(gradient*self.vertex_normals,axis=1))[:,None]*self.vertex_normals
                gradient[~active]=0
                self._last_eval=(self.points,indices,epsilon,cost,gradient.copy(),determinant)
            energy=float(cost.mean())
            q,alphas=gradient.copy(),[]
            for s,y,rho in reversed(self.history):
                alpha=rho*float(np.sum(s*q));alphas.append(alpha);q-=alpha*y
            if self.history:
                s,y,_=self.history[-1]
                h=float(np.sum(s*y)/max(np.sum(y*y),1e-300))
            else:
                h=float(np.median(self.length[active])**2*len(indices)*.05)
            r=q*h
            for (s,y,rho),alpha in zip(self.history,reversed(alphas)):
                r+=s*(alpha-rho*float(np.sum(y*r)))
            amount=-np.expm1(-12*min(max(dt,0),.1)*strength*pressure/max(1,passes))
            primary=-r*weights[:,None]
            accepted=False
            for attempt in range(2):
                direction=primary if attempt==0 else -gradient*self.length[:,None]**2*weights[:,None]
                if np.sum(direction*gradient)>=0:
                    continue
                direction=direction*amount
                cap=.15*amount*self.length*weights
                ratios=cap/np.maximum(np.linalg.norm(direction,axis=1),1e-300)
                direction*=min(1.,float(ratios[active].min()))
                if np.max(np.linalg.norm(direction,axis=1)/np.maximum(self.length,1e-30))<1e-11:
                    continue
                for trial in range(12):
                    candidate=self.points+direction*.5**trial
                    if project is not None:
                        candidate=np.ascontiguousarray(project(candidate,active),dtype=float)
                    candidate[~active]=self.points[~active]
                    actual=candidate-self.points
                    distance=np.linalg.norm(actual,axis=1)
                    # Reprojection can exceed the initial tangent proposal. Its
                    # actual displacement must also obey a hard local bound.
                    if not np.isfinite(candidate).all() or np.any(distance>.2*self.length+1e-12):
                        continue
                    after,new_gradient,end=self.metric(candidate,indices,epsilon,True)
                    new_energy=float(after.mean())
                    slope=float(np.sum(gradient*actual))
                    improves=new_energy<energy-max(1e-13,abs(energy)*1e-13)
                    if improves and (slope>=0 or new_energy<=energy+1e-4*slope) and self._path_safe(candidate,indices,determinant,end):
                        new_gradient-=(np.sum(new_gradient*self.vertex_normals,axis=1))[:,None]*self.vertex_normals
                        new_gradient[~active]=0
                        y=new_gradient-gradient
                        sy=float(np.sum(actual*y))
                        if sy>1e-12*np.linalg.norm(actual)*np.linalg.norm(y):
                            self.history.append((actual.copy(),y,1/sy))
                            self.history=self.history[-5:]
                        self.points=candidate
                        self._last_eval=(self.points,indices,epsilon,after,new_gradient.copy(),end)
                        determinant=end
                        moved+=int(np.count_nonzero(distance>self.length*1e-10))
                        backtracks+=trial
                        accepted=True
                        state="Relaxing"
                        break
                if accepted:
                    break
            if not accepted:
                self.history.clear()
        bad=int(np.count_nonzero(determinant<=1e-10))
        if bad:
            state="Repairing" if moved else "Repair constrained"
        self.last=dict(moved=moved,backtracks=backtracks,accepted=moved>0,bad_corners=bad,state=state)
        return self.last
