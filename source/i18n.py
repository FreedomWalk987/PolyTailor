# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Traditional Chinese interface text. English is the source language."""

import bpy

# Only strings unique to this add-on are listed; common words such as
# "Radius" or "Move" already come from Blender's own translation and adding
# them here would override it everywhere.
ZH_HANT = {
    "Stroke Transform Sensitivity": "筆劃變形靈敏度",
    "How fast Ctrl+move rotates / scales a stroke or strip preview (Shift: x0.1 finer)": "Ctrl＋移動旋轉／縮放筆劃或條帶預覽的速度（Shift：×0.1 更細）",

    "Strip folds at a bend; reduce width or redraw the curve": "條帶在彎折處重疊；請縮小寬度或重畫曲線",
    "Tight bend: inner side narrowed": "急彎：內側已收窄",
    "Merging the far end would collapse the quad strip": "末端合併會使四邊面條帶塌陷",
    # Build: face type, row strip from an edge chain
    "Mixed triangles and quads: build a triangle, click again to turn it into a quad": "三角與四邊混合：先建立三角面，再點一次轉成四邊面",
    "Quads only: build quad strips from the selected edge": "只建四邊面：從選取邊建立四邊面條帶",
    "The chain already has faces on both sides: select a chain of open edges": "邊鏈兩側都已有面：請選取開放邊組成的邊鏈",
    "Edge chain too long for a quad strip": "邊鏈太長，無法建立四邊面條帶",
    "Edge chain has overlapping points": "邊鏈有重疊的點",
    "A chain edge runs along the stroke; draw away from the chain": "邊鏈中有邊與筆劃同向；請朝離開邊鏈的方向畫",
    "Strip overlaps itself (highlighted); you can still add it": "條帶自身重疊（已標示）；仍可建立",
    "Merging the far row would collapse the quad strip": "末端整列合併會使四邊面條帶塌陷",
    "Stroke Density": "筆劃密度",
    "Ctrl+Wheel: change choice | Ctrl+LMB: fill | Esc: cancel": "Ctrl＋滾輪：換方案｜Ctrl＋左鍵：填補｜Esc：取消",
    "Add an empty mesh and start Build: in 3D mode at the 3D Cursor, or in Topo mode on the active object":
        "新增空白網格並開始建立：3D 模式下放在 3D 游標處，或重拓撲模式下放在作用中物件上",
    "Even": "均勻",
    "Fewer Poles": "少極點",
    "Follows Outline": "沿外框",
    "Fill {panels} panels | {seams} shared seams | {faces} faces": "填補 {panels} 個面板｜{seams} 條共用接縫｜{faces} 個面",

    "Field": "方向場",
    "Searching field candidates...": "搜尋方向場候選中…",
    "Computing...": "計算中…",
    "Search": "搜尋",
    "Fast search": "快速搜尋",
    "Full search": "完整搜尋",
    "Try the most likely layouts first; search fully only if the result is poor. Odd outlines always search fully":
        "先試最可能的布局，結果不佳時才完整搜尋；奇數外框一律完整搜尋",
    "Try every layout every time (slower)": "每次都嘗試所有布局（較慢）",
    "Better result found": "找到更好的結果",
    "This path cannot split the face": "此路徑無法切分這個面",
    "No pattern fits these points": "目前布點沒有合適的樣板",
    "The outline changed; preview again": "外框已改變，請重新預覽",
    "No fill found in time": "時間內找不到合適的填補",
    "Odd number of edges: quads need an even count": "奇數邊：四邊面需要偶數邊",
    "Outline too long to preview (max 256 edges)": "外框太長，無法預覽（上限 256 條邊）",
    "Remove the protection from the selected vertices": "解除選取頂點的保護",
    "Let the selected chains bend again": "解除選取鏈的剛性",
    "Provisional": "暫定",
    "Computing; click when the first result is shown": "計算中；顯示第一個結果後再點擊",
    "Computing; wheel when the first result is shown": "計算中；顯示第一個結果後再用滾輪",
    "Searching for a fill...": "搜尋填補方式中…",
    "Improving fill preview...": "正在改善填補預覽…",
    "Build": "建立",
    "Modeling Mode": "建模模式",
    "Switch 3D/Topo; Topo from a Blender tool switches to Build":
        "切換 3D／重拓撲；在 Blender 工具中選重拓撲會切到建立工具",
    "Change settings in the W wheel; Esc or a click outside closes it": "在 W 輪盤中調整設定；Esc 或點外部關閉",
    "Alt+Wheel Smooth settings (Build and Fill)": "Alt＋滾輪平滑設定（建立與填補）",
    "Slide the selection along its edges": "沿邊滑動選取",
    "Action unavailable for the current selection":"目前的選取無法執行此操作",
    "Enter a valid number":"請輸入有效數字",
    "Wheel: scroll options":"滾輪：捲動選項",
    "Enter: confirm | Esc: cancel":"Enter：確認｜Esc：取消",
    "The outline crosses a UV seam; unwrap the new faces": "外框跨過 UV 縫線，請重新拆解新面的 UV",
    "Part of the outline has no faces next to it; unwrap the new faces": "部分外框旁沒有面，請重新拆解新面的 UV",
    "Adaptive Partition": "自動分區",
    "Continue Grid": "延續網格",
    "Auto Pattern": "自動選樣板",
    "Four-sided Grid": "四邊格線",
    "Triangle Pattern": "三邊樣板",
    "Pentagon Pattern": "五邊樣板",
    "Diagonal Transition": "斜向過渡",
    "Aligned Transition": "對齊過渡",
    "U-turn Transition": "迴轉過渡",
    "Two-corner Pattern": "雙角樣板",
    "One-corner Pattern": "單角樣板",
    "Disk Pattern": "圓盤樣板",
    "No valid pattern for this point plan": "目前布點沒有合適的樣板",
    "Preview boundary limit (256 points)": "預覽上限為 256 個外框點",
    "Fill corners no longer available": "填補角點已失效",
    "Invalid boundary": "外框無效",
    "points": "點",
    "segments": "段",
    "segment": "段",
    "edges": "邊",
    "Odd count": "奇數邊：四邊面需要偶數邊",
    # tools
    "Push vertices around with a soft brush": "用柔邊筆刷推移頂點",
    # brush settings
    "Brush radius in pixels": "筆刷半徑（像素）",
    "How strongly the brush acts": "筆刷作用的強弱",
    "0 = hard edge, 1 = smooth fade toward the rim": "0 = 硬邊，1 = 往外緣平滑淡出",
    "Which vertices the brushes may affect, based on selection": "依選取狀態決定筆刷可影響的頂點",
    "Affect selected and unselected vertices": "影響已選與未選頂點",
    "Leave selected vertices alone": "不動已選頂點",
    "Affect only selected vertices": "只影響已選頂點",
    "How brushes treat vertices on the mesh border": "筆刷如何處理網格邊界上的頂點",
    "Boundary vertices move freely": "邊界頂點自由移動",
    "Boundary vertices slide along the border": "邊界頂點沿邊界滑動",
    "Boundary vertices stay in place": "邊界頂點固定不動",
    "Brushes also affect vertices hidden behind the surface": "筆刷也影響被表面遮住的頂點",
    "Affect Hidden Vertices": "影響被遮擋的點",
    # operators
    "Move the selection or the hovered element; box select from empty space": "拖動選取或滑鼠下的元素；從空白處開始則框選",
    "Adjust Brush": "調整筆刷",
    "Drag to change the brush radius, strength or falloff": "拖曳以調整筆刷半徑、力量或衰減",
    "Stroke": "筆劃",
    "Corner Sensitivity": "轉角靈敏度",
    "Minimum turn angle for a fixed stroke corner; lower angles detect more corners": "保留固定轉角頂點的最小轉向角度；角度越小，偵測越多轉角",
    "Release: stroke preview": "放開左鍵：筆劃預覽",
    "Ctrl+move: rotate / scale | Wheel: segments | LMB / Enter: add | Esc / RMB: cancel": "Ctrl＋移動：旋轉／縮放｜滾輪：段數｜左鍵／Enter：建立｜Esc／右鍵：取消",
    # Build
    # preferences
    "Pick Radius": "點選範圍",
    "Screen distance in pixels for picking vertices and edges": "點選頂點與邊的螢幕距離（像素）",
    "Accent Color": "強調色",
    "Relax topology; hold Ctrl to temporarily drag with the same brush, Alt for Grid Relax": "放鬆拓撲；按住 Ctrl 暫時以目前筆刷拖移，按住 Alt 為格線放鬆",
    "Robust Relax (Alt: Grid Relax)": "穩健放鬆（Alt：格線放鬆）",
    "Snapping": "吸附",
    "Reset Brush": "重設筆刷",
    "Ignore Derived Objects": "忽略衍生物件",
    "shared mesh": "共用網格",
    "Snap to Itself": "吸附自身",
    "Vertex / Edge / Edge Center snapping also uses the mesh being edited (not the vertices being moved)":
        "頂點／邊／邊線中心吸附也使用正在編輯的網格（不含正在移動的點）",
    "Use Blender Snapping": "使用原生吸附",
    "Left-drag, G and E use Blender's own move for Vertex / Edge / Edge Center snapping (Ignore Derived Objects does not apply). Build placement keeps PolyTailor's snapping":
        "左鍵拖曳、G、E 的頂點／邊／邊線中心吸附改用 Blender 原生移動（「忽略衍生物件」不適用）。建立工具放點仍用 PolyTailor 吸附",
    "Snap Distance": "吸附距離",
    "Vertex Snap Limit": "頂點吸附的頂點數上限",
    "Objects with more vertices than this (e.g. 3D scans) are not used for Vertex / Edge / Edge Center snapping, which would stall while indexing them. Surface snapping still uses them":
        "點數超過此值的物件（例如 3D 掃描）不參與頂點／邊／邊線中心吸附，避免建立索引時卡住；表面吸附仍會使用",
    "Sources": "來源物件",
    "Display & Mirror": "顯示與鏡像",
    "Defaults": "預設值",
    "Shortcuts": "快捷鍵",
    "Snap": "吸附",
    "Auto Merge": "自動合併",
    "Save as Default": "存為預設",
    "Use the current PolyTailor settings - tool and brush settings, snapping, Stop at Edge, Auto Merge - as the defaults for new files and scenes":
        "把目前的 PolyTailor 設定（工具與筆刷、吸附、邊緣限制、自動合併）存為新檔案與新場景的預設",
    "Restore Factory Defaults": "還原出廠預設",
    "Forget the saved defaults and put this scene's PolyTailor settings back to PolyTailor's own defaults":
        "清除已儲存的預設，並把此場景的 PolyTailor 設定還原成 PolyTailor 原本的預設",
    "Using your saved defaults": "使用你儲存的預設",
    "Using PolyTailor's defaults": "使用 PolyTailor 出廠預設",
    "PolyTailor defaults saved": "已儲存 PolyTailor 預設",
    "PolyTailor factory defaults restored": "已還原 PolyTailor 出廠預設",
    "Could not save the defaults": "無法儲存預設",
    "Could not remove the saved defaults": "無法清除已儲存的預設",
    "Screen distance in pixels for Vertex / Edge / Edge Center snapping (Blender uses 30)":
        "頂點／邊／邊線中心吸附的螢幕距離（像素，Blender 原生為 30）",
    "modifier": "修改器",
    "Do not snap to objects generated from the mesh being edited: objects sharing its mesh data, or whose modifiers / geometry nodes read it (e.g. a projected copy)":
        "不吸附由正在編輯的網格產生的物件：共用同一份網格資料，或修改器／幾何節點讀取它的物件（例如投影出來的副本）",
    "Keep split vertices on their original edges while placing": "放置時讓切分出的頂點保持在原本的邊上",
    "Preserve quad junctions when cutting adjacent edges": "切割相鄰邊時保留四邊形交會處",
    "Move selection on the surface; G again slides; E again offsets an extrusion": "在表面上移動選取；再按 G 滑動；擠出時再按 E 偏移",
    "Only Selected": "僅已選取",
    "Unselected": "未選取",
    "Relax Selection": "放鬆選取",
    "Topo": "重拓撲",
    "Model freely without snapping to other objects": "自由建模，不吸附其他物件",
    "Retopologize using the Topo defaults": "使用重拓撲預設的吸附與合併設定",
    "Brush Relax": "筆刷放鬆",
    "Relax": "放鬆",
    "Relax / Tweak": "放鬆／調校",
    "Relax Algorithm": "放鬆演算法",
    "Tweak": "調校",
    "Native": "原生",
    "Grid Relax": "格線放鬆",
    "Use the Relax tool's Grid Relax (Alt): even, square quads and straight edge lines": "使用放鬆工具的格線放鬆（Alt）：邊長平均、四邊形方正、線圈拉直",
    "Even out edge lengths, square up quads and straighten edge lines": "邊長拉平均、四邊形拉方正、線圈拉直",
    "Relax passes per brush step": "每次筆刷步驟的放鬆次數",
    "Even Out Spacing": "均勻間距",
    "Even out uneven spacing along each row of quads while keeping row widths (holding loops, density changes); off: keep every quad's own size and proportions": "沿每一排四邊面把不均勻的間距調平均，同時保留每排的寬度（卡線、疏密變化）；關閉：保留每個四邊面原本的大小和比例",
    "LoopTools Relax": "LoopTools 放鬆",
    "Use Blender's Smooth Vertices": "使用 Blender 的平滑頂點",
    "Use the same relaxation as the Relax brush": "使用與放鬆筆刷相同的放鬆",
    "Use the embedded LoopTools Relax algorithm; no extension required": "使用內建的 LoopTools 放鬆演算法，不需另外安裝擴充套件",
    "Active Tool": "目前工具",
    "Tri/Quad": "三角／四邊",
    "along global": "沿全域軸",
    "Restore the default brush radius, strength and falloff (Relax also restores its algorithm settings)":
        "恢復預設的筆刷半徑、力量與衰減（放鬆也會恢復演算法設定）",
    "Brush:": "筆刷：",
    "Selected:": "選取：",
    "Boundary:": "邊界：",
    "Stop at Edge": "邊緣限制",
    "With surface snapping on, points moved past the edge of the source stop at the edge instead of leaving the surface":
        "開啟表面吸附時，拉出來源物件範圍的點會停在模型邊緣，而不是落到面向畫面的平面上",
    "New PolyTailor Mesh": "新增 PolyTailor 網格",
    "New Mesh (3D)": "新網格（3D）",
    "Retopology Mesh (Topo)": "重拓撲網格",
    "At 3D Cursor": "在 3D 游標位置",
    "From Active Object": "沿用作用中物件",
    "Location": "位置",
    "No active object": "沒有作用中的物件",
    "Place the new object at the 3D Cursor": "把新物件放在 3D 游標位置",
    "Copy the active object's location, rotation and scale": "沿用作用中物件的位置、旋轉與縮放",
    "Protect the selected vertices from Tweak, Relax and Alt+Wheel": "保護選取的頂點，不受調校、放鬆與 Alt＋滾輪影響",
    "Constraint Tools": "固定工具",
    "Keep each connected selected chain rigid during Relax, Tweak and Alt+Wheel; the whole chain can still move and rotate": "選取的相連鏈在放鬆、調校與 Alt＋滾輪時保持剛性，整條仍可移動與旋轉",
    "Select a connected edge chain to Glue": "請選取相連的邊鏈以黏合",
    "Select a Glue member to unglue": "請選取黏合成員以解除整條黏合",
    "Finish the active transform first": "請先完成目前的變換操作",
    "Preview Fill": "預覽面填色",
    "Preview Border": "預覽線條顏色",
    "Icon Colors": "圖示顏色",
    "Custom": "自訂",
    "Icon Lines": "圖示線條",
    "Icon Secondary": "圖示次要色",
    "Icon Accent": "圖示強調色",
    "Colors of the tool icons in the W wheel": "W 輪盤上工具圖示的顏色",
    "Follow the Blender theme like its own icons: the theme's text color, the Accent Color, and the theme's icon saturation, alpha and border": "像 Blender 原生圖示一樣跟著主題變色：主題文字色、強調色，並套用主題的圖示飽和度、透明度與外框",
    "PolyTailor's own colors, set below; the theme does not change them": "PolyTailor 自己的配色，可在下方修改，不受主題影響",
    "Quad Stability": "四邊面穩定度",
    "Quad mode: how much new quads fan out when the cursor is off to one side; at 1 the far edge stays square to the cursor direction (least fan-out)":
        "四邊形模式：游標偏向一側時新四邊面向外展開的程度；為 1 時遠端邊與游標方向垂直（展開最少）",
    "Constrain": "限制在線上",
    "Junctions": "四邊面交接",
    "Move on Surface": "沿表面移動",
    "Slide on Surface": "沿表面滑動",
    "PolyTailor Wheel": "PolyTailor 輪盤",
    "Open the PolyTailor wheel: tools and settings": "開啟 PolyTailor 輪盤：工具與設定",
    "Retopology Settings": "重拓撲設定",
    "Source Object": "來源物件",
    "Source Collection": "來源集合",
    "Selected Sources Only": "僅使用已選取來源",
    "Selectable Sources Only": "僅使用可選取來源",
    "Add Mirror": "新增鏡像",
    "Add a Mirror modifier with Clipping and Merge on": "新增開啟截割與合併的鏡像修改器",
    "Alt+Wheel Smooth": "Alt＋滾輪平滑",
    "Alt+Wheel relaxes the selection; wheel down steps back": "Alt＋滾輪放鬆選取；向下滾動回復上一步",
    "Automatic Retopology Setup": "自動設定重拓撲環境",
    "Set up surface snapping and the overlay while a PolyTailor tool is active; restore on exit": "使用 PolyTailor 時設定表面吸附與疊層，離開工具時還原",
    "Show Wheel Hint": "顯示輪盤提示",
    "Dense Relax Feedback": "高面數放鬆即時顯示（實驗）",
    "Experimental local display for dense 3D Relax in Solid retopology views; full native display synchronizes on release": "高面數 3D 放鬆時使用局部顯示；限實體重拓撲視圖，放開後同步完整原生顯示",
    "Model instead of retopologize: no snapping to other objects, no Auto Merge and no Stop at Edge in any PolyTailor tool; Relax keeps the mesh on its own shape. Turn off for the Topo defaults":
        "改為建模而非重拓撲：所有 PolyTailor 工具都不吸附其他物件、不自動合併、不做邊緣限制；放鬆讓網格維持自己的形狀。關閉後套用重拓撲預設設定",
    'PolyTailor Settings': 'PolyTailor 設定',
    "This object's transform has a zero scale; fix the scale of the object or its parents": '這個物件的變換有縮放為零的情況；請修正物件或其父物件的縮放',
}


ZH_HANT.update({
    "O: proportional editing | Shift+O: falloff": "O：比例化編輯｜Shift＋O：切換衰減",
    "Proportional editing: {falloff} | Wheel: influence radius": "比例化編輯：{falloff}｜滾輪：調整影響半徑",
    "Proportional size": "比例化編輯半徑",
    "Boundary offset | LMB / Enter: confirm | Esc / RMB: cancel": "邊界偏移｜左鍵／Enter：確認｜Esc／右鍵：取消",
    "Hold Ctrl: preview | Ctrl+LMB: add | Drag: place new geometry": "按住 Ctrl：預覽｜Ctrl＋左鍵：建立｜拖曳：調整新位置",
    "LMB drag: move, or box select from empty space | G: move | E: extrude": "左鍵拖曳：移動／空處框選｜G：移動｜E：擠出",
    "F: radius | Shift+F: strength | Ctrl+F: falloff": "F：半徑｜Shift＋F：力量｜Ctrl＋F：衰減",
    "LMB drag: move vertices under the brush": "左鍵拖曳：推移筆刷範圍內的頂點",
    "Nothing to add here": "目前位置無法建立",
    "Add vertex": "建立頂點",
    "Extend edge": "延伸邊",
    "Add triangle": "建立三角面",
    "Add quad": "建立四邊面",
    "Extend triangle to quad": "將三角面延伸為四邊面",
    "Bridge edges": "橋接邊",
    "Split edge": "分割邊",
    "Connect and split edge": "連接並分割邊",
    "Cut face": "切割面",
    "Cut through face center": "穿過面中心切割",
    "Cut across face edges": "跨越面邊切割",
    "Connect vertex to edge": "將頂點連接至邊",
    "Close wire into face": "封閉線段建立面",
    "Drag: place | Shift: precision | Release LMB: confirm | Esc / RMB: cancel": "拖曳：放置｜Shift：精細調整｜放開左鍵：確認｜Esc／右鍵：取消",
    "Preview: {action}": "預覽：{action}",
    "Temporary Tweak | Release Ctrl: resume Relax | Uses the Relax brush settings": "暫時調校｜放開 Ctrl：繼續放鬆｜使用放鬆筆刷的設定",
    "Release LMB: finish | Esc / RMB: cancel the entire stroke": "放開左鍵：完成｜Esc／右鍵：取消整段筆劃",
    "{label}: {value} | LMB / Enter: confirm | Esc / RMB: cancel": "{label}：{value}｜左鍵／Enter：確認｜Esc／右鍵：取消",
    "LMB / Enter: confirm | Esc / RMB: cancel": "左鍵／Enter：確認｜Esc／右鍵：取消",
    "Release LMB: confirm | Esc / RMB: cancel": "放開左鍵：確認｜Esc／右鍵：取消",
    "X / Y / Z: global axis (again: unlock) | Shift: precision": "X／Y／Z：全域軸，再按解除｜Shift：精細移動",
    "Global axis: {axis}": "全域軸：{axis}",
    "E again: boundary offset": "再次按 E：邊界偏移",
    "Wheel: extrusion cuts": "滾輪：調整擠出切段",
    "G again: slide along edges": "再次按 G：沿邊滑動",
    "Slide along edges | LMB / Enter: confirm | Esc / RMB: cancel": "沿邊滑動｜左鍵／Enter：確認｜Esc／右鍵：取消",
    "Alt+Wheel up: relax selection | Wheel down: step back | Release Alt: confirm": "Alt＋滾輪向上：放鬆選取｜向下：退回一步｜放開 Alt：確認",
    "Temporary Tweak": "暫時調校",
    "W: tools / settings": "W：工具與設定",
    "Select Mode": "選取模式",
    "Boundary: {boundary} | Radius: {radius} px | Strength: {strength} | Falloff: {falloff}": "邊界：{boundary}｜半徑：{radius} px｜力量：{strength}｜衰減：{falloff}",
    "Fill": "填補",
    "Pattern: {name}": "樣板：{name}",
    "Odd edge count: one triangle": "奇數邊：會留一個三角面",
    "Loop too short": "線圈太短",
    "Loop too long to fill": "線圈太長，無法填補",
    "No pattern found in time": "時間內找不到合適的樣板",
})


ZH_HANT.update({
    "Hold Ctrl: preview the selected outline / hovered hole | Ctrl+LMB: fill":
        "按住 Ctrl：預覽已選外框／滑鼠下的洞｜Ctrl＋左鍵：填補",
    "Hover a hole, or select one closed outline": "移到洞上，或選取一個封閉外框",
    "Select only open edges (at most one face each)": "請只選取開放邊（每條最多連接一個面）",
    "Close the selected outline before filling": "選取的線段尚未封閉，請先接成一圈再填補",
    "Select one closed loop without branches": "請選取單一封閉線圈，不可有分岔或多個圈",
    "Start": "起點",
    "This outline encloses no area; select another one": "這個外框圍不出面積，請改選其他外框",
    "Invalid fill geometry": "填補幾何無效",
    "This layout folds; try another choice (Ctrl+Wheel)": "此布局會折疊，請換一個選擇（Ctrl＋滾輪）",
    "Invalid fill connectivity": "填補連接關係無效",
    "This loop is already filled": "這個線圈已經有面",
    "The faces around the outline have inconsistent normals; recalculate normals first": "外框旁的面法線方向不一致，請先重新計算法線",
    "This fill would give an edge more than two faces": "填補會讓同一條邊連接超過兩個面",
})


ZH_HANT.update({
    # terminology
    "Pin": "釘選",
    "Glue": "黏合",
    "Sharps": "銳邊",
    "Creases": "摺痕",
    "Faces": "面型",
    "Faces: {faces}": "面型：{faces}",
    "Faces built from a selected edge": "從選取邊建立的面型",
    "Merge Radius": "合併半徑",
    "Auto Merge radius in screen pixels": "以螢幕像素計算的自動合併半徑",
    "Stop at Edge": "邊緣限制",
    "Targets": "吸附目標",
    "Retopo Shading": "重拓撲著色",
    "Relax Algorithm": "放鬆演算法",
    "Alt+Wheel Smooth": "Alt＋滾輪平滑",
    "Version {version}": "版本 {version}",
    "Input": "輸入",
    "Display": "顯示",
    "Colors": "顏色",
    "Classified": "分類",
    "Cut": "切割",
    "Single patch": "單片",
    "Show Debug Info": "顯示除錯資訊",
    "Show the internal quality numbers of each fill choice and keep a debug log (newfill_log.jsonl) in the add-on's user folder":
        "顯示每個填補選擇的內部品質數值，並在附加元件的使用者資料夾保存除錯紀錄（newfill_log.jsonl）",
    "Extended {sides}-sided Pattern": "擴充{sides}邊樣板",
    "Robust Relax (Alt: Grid Relax)": "穩健放鬆（Alt：格線放鬆）",
})


# Visible strings that had no zh-TW
ZH_HANT.update({
    "Switch Tool": "切換工具",
    "Alt+Wheel: relax": "Alt＋滾輪：放鬆",
    "Invalid triangle site": "三角形位置無效",
    "Triangle site changed; preview again": "三角形位置已改變，請重新預覽",
    "Panel network would create a non-manifold seam": "面板網路會讓接縫邊連接超過兩個面",
    "Panel selection changed; preview again": "面板選取已改變，請重新預覽",
    "Select closed empty panels without dangling connecting edges": "請選取封閉的空白面板，且不可有懸空的連接邊",
    "Merge the overlapping seam vertices first": "請先合併重疊的接縫頂點",
    "PolyTailor Wheel error; see console": "PolyTailor 輪盤錯誤，請查看主控臺",
})


ZH_HANT.update({
    # plain guard messages (ops/common.py)
    "Another tool is editing this mesh; finish it first": "另一個工具正在編輯此網格，請先完成",
    "The mesh being edited changed; the tool stopped": "正在編輯的網格已改變，工具已停止",
    "Another tool is editing this mesh; stopped and kept the changes": "另一個工具正在編輯此網格，已停止並保留變更",
    # one Fill failure message for the technical guard cases (core/fill.py user_text)
    "This outline cannot be filled; try another selection": "這個外框無法填補，請改選其他範圍",
    # Build line preview row without the segment count (ui/hud.py)
    "Wheel: segments | Ctrl+LMB: add | Drag: place": "滾輪：段數｜Ctrl＋左鍵：建立｜拖曳：放置",
    # tool names capitalised, plain verbs lower case
    "Hold LMB: Relax | Hold Alt+LMB: Grid Relax | Hold Ctrl and drag: temporary Tweak":
        "按住左鍵：放鬆｜按住 Alt＋左鍵：格線放鬆｜按住 Ctrl 拖曳：暫時調校",
    "Hold Ctrl to preview, Ctrl+LMB to add": "按住 Ctrl 預覽，Ctrl＋左鍵建立",
    # more visible strings
    "Stroke segments overlap here": "筆劃的分段在此重疊",
    "Value must be finite": "數值必須是有限數",
    "Box select, adding to the selection": "框選並加入目前的選取",
    "Segments per screen length: 4 gives one segment every 80 pixels; bends add more. The model's size does not matter":
        "依螢幕長度分段：4 表示每 80 像素一段，彎曲處會加段；與模型大小無關",
    "Track Brush Cursor": "追蹤筆刷游標",
})


# Terms that Blender translates differently in its default context: registered only in
# the add-on context.
TERMS = {
    "Build": "建立",
    "Fill": "填補",
    "Relax": "放鬆",
    "Tweak": "調校",
    "Stroke": "筆劃",
    "Off": "解除",
    "Pin": "釘選",
    "Glue": "黏合",
    "Move": "移動",
    "Slide": "滑動",
    "Lock": "鎖定",
    "Input": "輸入",
    "Display": "顯示",
    "Colors": "顏色",
}

CONTEXT = "Poly Toolset"
VIEWPORT_HELP_CONTEXT = "Poly Toolset Viewport Help"   # Off = 關閉 here, 解除 for Pin / Glue
VIEWPORT_HELP = {"Viewport Help": "視窗說明", "Full": "完整", "Compact": "精簡", "Off": "關閉"}
SHORT_TIPS = ()   # tooltips under four words (descriptions use "*")

# Blender's own words shown on add-on surfaces under CONTEXT; the text is Blender 5.2's own zh_HANT.
BLENDER_WORDS = {
    'Radius': '半徑',
    'Strength': '力量',
    'Falloff': '衰減',
    'Drag': '拖曳',
    'Extend': '擴展',
    'Deselect': '取消選取',   # Blender 5.2 zh_HANT (blender.mo)
    'Brush': '筆刷',
    'Keymap': '按鍵對應',
    'Corners': '邊角',
    'Seams': '縫線',
    'Iterations': '迭代',
    'All': '全部',
    'Selected': '所選項',
    'Boundary': '邊界',
    'Axis': '軸',
}


def tr(message, context="*"):
    """Translate with the add-on's own terms first, then Blender's context."""
    if context not in ("*", CONTEXT):   # a specific add-on context (e.g. Viewport Help: Off = 關閉) wins
        text = bpy.app.translations.pgettext_iface(message, context)
        if text != message:
            return text
    text = bpy.app.translations.pgettext_iface(message, CONTEXT)
    return text if text != message else bpy.app.translations.pgettext_iface(message, context)


def tr_rpt(message, context="*"):
    """tr for operator reports: follows Blender's Reports translation setting (tr follows Interface)."""
    rpt = bpy.app.translations.pgettext_rpt
    if context not in ("*", CONTEXT):
        text = rpt(message, context)
        if text != message:
            return text
    text = rpt(message, CONTEXT)
    return text if text != message else rpt(message, context)


def friendly_error(exc, fallback):
    """the text for a caught exception. A raised text that has its own translation is shown as is; any
    other exception text goes to the console and the user sees `fallback` (an existing translated message)."""
    text = str(exc)
    if text in ZH_HANT:
        return text
    print(f"PolyTailor: {type(exc).__name__}: {text}")
    return fallback


def _build():
    """Translation tables: Traditional Chinese when Blender's interface language is Traditional Chinese."""
    table = {}
    # the add-on's text lives in its own context; Blender's own words keep Blender's text.
    # "*": sentences only (tooltips and reports: Blender translates them in the default context only).
    for src, dst in ZH_HANT.items():
        if len(src.split()) >= 4 or src in SHORT_TIPS:
            table[("*", src)] = dst
    # The toolbar draws tool labels in Blender's "Operator" context; the add-on's tool labels are message ids of
    # their own there (the word and a trailing space, tools.py TOOL_LABEL_IDS), so Blender's own words stay unchanged.
    for src in ("Build", "Fill", "Tweak", "Relax"):
        table[("Operator", src + " ")] = TERMS[src]
    for src, dst in VIEWPORT_HELP.items():
        table[(VIEWPORT_HELP_CONTEXT, src)] = dst
    for src, dst in (("Vertex", "點"), ("Edge", "線"), ("Face", "面")):
        table[("Poly Toolset Select Mode", src)] = dst
    # Build's face toggle: official names in their own context, so Blender's own
    # "Quad" elsewhere keeps its translation.
    for src, dst in (("Tri/Quad", "三角／四邊"), ("Quad", "四邊形"), ("Faces:", "面型："),
                     ("Faces", "面型"), ("Faces: {faces}", "面型：{faces}"),
                     ("Faces built from a selected edge", "從選取邊建立的面型")):
        table[("Poly Toolset Faces", src)] = dst
    for src, dst in (("Correct Face Attributes", "修正面屬性"),):
        table[("Poly Toolset Controls", src)] = dst
    # Add-on terms first: the add-on's own context, so a term such as "Build" reads
    # 建立 on PolyTailor surfaces while Blender's own "Build" (建置) stays unchanged.
    for src, dst in BLENDER_WORDS.items():   # Blender's own words on add-on surfaces, Blender's own text
        table[(CONTEXT, src)] = dst
    for src, dst in ZH_HANT.items():
        table[(CONTEXT, src)] = dst
    for src, dst in TERMS.items():
        table[(CONTEXT, src)] = dst
    return {"zh_HANT": table}


def apply():
    """Register the add-on's translations."""
    # Blender keeps this registry outside Python module globals. A reload or a
    # failed enable can leave the previous table alive under the same package.
    # Build first, then replace only our own entry; do not hide register errors.
    tables = _build()
    bpy.app.translations.unregister(__package__)
    bpy.app.translations.register(__package__, tables)
    for window in getattr(getattr(bpy.context, "window_manager", None), "windows", ()):
        for area in window.screen.areas:
            area.tag_redraw()


def register():
    apply()


def unregister():
    bpy.app.translations.unregister(__package__)


ZH_HANT.update({"Viewport Help": "視窗說明", "Full": "完整", "Compact": "精簡",
                "Key help in the viewport; keys are always listed in the status bar": "視窗中的按鍵說明；按鍵一律列在狀態列",
                "Shift+W: all settings": "Shift＋W：所有設定", "All Tools": "所有工具"})

ZH_HANT.update({
    "How widely Fill searches": "填補的搜尋範圍",
    "Searching wider...": "擴大搜尋中…",
    "Wider search": "擴大搜尋",
    "MMB Tool Switch": "中鍵切換工具",
    "Click MMB: Build \u2194 Brush; Ctrl+MMB: Build \u2194 Fill. Drags still orbit and zoom":
        "點中鍵：建立 \u2194 筆刷；Ctrl＋中鍵：建立 \u2194 填補。拖曳仍為旋轉與縮放",
    "Select one closed outline, or hover a hole, then hold Ctrl to preview.\nCtrl+Wheel: change choice, Ctrl+click: fill":
        "選取一個封閉外框，或移到洞上，再按住 Ctrl 預覽。\nCtrl＋滾輪：換方案，Ctrl＋左鍵：填補",
})

ZH_HANT.update({"Hold Ctrl: preview the selected outline or hovered hole, Ctrl+Wheel: change choice, Ctrl+click: fill": "按住 Ctrl：預覽已選外框或滑鼠下的洞，Ctrl＋滾輪：換方案，Ctrl＋左鍵：填補"})

ZH_HANT.update({"MMB: Brush / Build | Ctrl+MMB: Fill / Build": "中鍵：筆刷／建立｜Ctrl＋中鍵：填補／建立"})

ZH_HANT.update({"{faces} faces": "{faces} 個面"})

ZH_HANT.update({"Switch Tool: Brush": "切換工具：筆刷", "Switch Tool: Fill": "切換工具：填補"})

ZH_HANT.update({   # Drag box-select description, LoopTools Relax loop error (core/looptools_relax.py)
    "Box select, removing from the selection": "框選並從目前的選取中移除",
    "Select a connected loop of at least three non-coincident vertices": "請選取至少有三個不重疊頂點的相連線圈",
})

ZH_HANT.update({   # the Build stroke / strip and Relax fallback messages
    "This stroke could not be built here": "無法在此處建立這條筆劃",
    "Relax could not continue here": "放鬆無法在此繼續",
})

ZH_HANT.update({   # the shown wording of Fill refusals (core/fill._SHOWN)
    "Outline too long to fill (max 256 edges)": "外框太長，無法填補（上限 256 條邊）",
    "This outline is already filled": "這個外框已經填上面了",
    "Select one outline, or panels that share edges": "請選取一個外框，或共用邊的多個面板",
    "These panels would give a seam edge more than two faces": "這些面板會讓接縫邊連接超過兩個面",
})

ZH_HANT.update({   # Fill refusals: separate loops, multi-panel time-out
    "Select one hole, or panels that share edges": "請選取一個洞，或共用邊的多個面板",
    "Panel search took too long; select fewer panels": "面板搜尋時間過長，請少選一些面板",
})
