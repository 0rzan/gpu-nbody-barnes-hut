n = int(input())
ps = [tuple(map(float, input().split())) for _ in range(n)]
cnt_copy = 1000000
print((n - 1) * cnt_copy + 1)
x, y, z, vx, vy, vz, m = ps[0]
print(f"{x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f} {m:.6f}")
for i in range(cnt_copy):
    for x, y, z, vx, vy, vz, m in ps[1:]:
        print(f"{x:.6f} {y:.6f} {z+i*10:.6f} {vx:.6f} {vy:.6f} {vz:.6f} {m:.6f}")