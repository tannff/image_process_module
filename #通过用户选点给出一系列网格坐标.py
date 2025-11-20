import numpy as np
import time

# 生成直径70、高度70的圆柱表面坐标（包含侧面和上下盖），步长为1
radius = 35
height = 70
center = radius

# 构建 x,y 网格 (0..70)
xv, yv = np.meshgrid(np.arange(0, 2*radius + 1), np.arange(0, 2*radius + 1), indexing='xy')
dx = xv - center
dy = yv - center
dist = np.sqrt(dx**2 + dy**2)

# 侧面：距离接近半径的格点（环带）
lateral_mask = np.abs(dist - radius) <= 0.5
lateral_xy = np.column_stack((xv[lateral_mask], yv[lateral_mask]))  # (N_ring, 2)

# 上下盖：盘面内所有格点（包含边界）
disk_mask = dist <= radius + 0.5
disk_xy = np.column_stack((xv[disk_mask], yv[disk_mask]))  # (N_disk, 2)

coords_list = []

# 侧面 z = 1..height-1
zs_side = np.arange(1, height)
if lateral_xy.size and zs_side.size:
    lateral_all = np.tile(lateral_xy, (zs_side.size, 1))                      # 每个 z 重复一遍环上所有点
    zs_rep = np.repeat(zs_side, repeats=lateral_xy.shape[0])                  # 对应的 z 列
    side_coords = np.column_stack((lateral_all, zs_rep.astype(int)))
    coords_list.append(side_coords)

# 底面 z = 0 与 顶面 z = height
if disk_xy.size:
    bottom_coords = np.column_stack((disk_xy, np.zeros(disk_xy.shape[0], dtype=int)))
    top_coords = np.column_stack((disk_xy, np.full(disk_xy.shape[0], height, dtype=int)))
    coords_list.append(bottom_coords)
    coords_list.append(top_coords)

if coords_list:
    grid_coords = np.vstack(coords_list).astype(int)
else:
    grid_coords = np.empty((0, 3), dtype=int)

print("圆柱表面点总数：", grid_coords.shape[0])
print(grid_coords)

#通过用户选点给出一系列网格坐标和胶囊当前位置的网格坐标
sample_idx = np.random.choice(grid_coords.shape[0], size=10, replace=False)
sampled_points = grid_coords[sample_idx]
print("随机选取的10个网格坐标：")
print(sampled_points)

#胶囊当前位置
capsule_pos = np.array([[25, 25, 25]])  # 假设胶囊当前位置在 (25, 25, 25)
print(capsule_pos)

#标记区域中的异变点
#加一个计时器
start_time = time.time()

#给当前胶囊和上述各个随机点坐标生成邻接矩阵
#计算胶囊和上述十个选点两两之间的曼哈顿距离，并生成邻接矩阵
points = np.vstack((capsule_pos, sampled_points))           # shape (11, 3)，第1行/列为胶囊
manhattan_matrix = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)  # shape (11, 11)
print("曼哈顿距离矩阵（行/列顺序：capsule + sampled_points）：")
print(manhattan_matrix)

#生成节点序列矩阵
#node_sequence = np.tile(np.arange(1, 12, dtype=int), (11, 1))  # shape (11, 11)
#print("node_sequence shape:", node_sequence.shape)
#print(node_sequence)

#贪心算法寻找路径
n = points.shape[0]
visited = np.zeros(n, dtype=bool)
visited[0] = True  #起始点为胶囊当前位置
current_idx = 0
path_indices = [current_idx]
path_positions = [points[current_idx]]

while visited.sum() < n:
    dists = manhattan_matrix[current_idx].astype(float).copy()  #确保为浮点，能承载 np.inf
    dists[visited] = np.inf  #忽略已访问节点

    finite_mask = np.isfinite(dists)
    if not np.any(finite_mask):
        #没有可达的未访问节点，退出循环
        break

    #只在有限距离中选最小值，避免将 np.inf 转为整数时发生溢出错误
    candidate_idxs = np.where(finite_mask)[0]
    rel_idx = int(np.argmin(dists[candidate_idxs]))
    next_idx = int(candidate_idxs[rel_idx])

    visited[next_idx] = True
    current_idx = next_idx
    path_indices.append(current_idx)
    path_positions.append(points[current_idx])

print("最终访问顺序索引（points 中的位置）：", path_indices)

#写出胶囊到以上点的具体路径
def generate_detailed_path(path_positions):
    detailed_path = []
    for i in range(1, len(path_positions)):
        start = path_positions[i - 1]
        end = path_positions[i]
        move = end - start
        detailed_path.append(f"x->{move[0]}, y->{move[1]}, z->{move[2]}")
    return detailed_path
detailed_path = generate_detailed_path(path_positions)

#打印访问的坐标和具体路径
for i, step in enumerate(detailed_path):
    print(f"访问坐标：{path_positions[i+1]}, 移动路径：{step}")

print("移动总的路径长度：", sum(manhattan_matrix[path_indices[:-1], path_indices[1:]].flatten()))

#结束计时,计时为ms级别，显示单位为ms
end_time = time.time()
elapsed_time = (end_time - start_time) * 1000  #转为毫秒
print(f"程序运行时间: {elapsed_time:.2f} ms")