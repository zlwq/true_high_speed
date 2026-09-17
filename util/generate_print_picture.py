import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rows = 5
cols = 10

fig, ax = plt.subplots(figsize=(13, 6.6))

for y in range(rows):
    for x in range(cols):
        cx = x + 0.5
        cy = rows - y - 0.5

        outer = Circle((cx, cy), 0.38, color='black')
        inner = Circle((cx, cy), 0.24, color='white')

        ax.add_patch(outer)
        ax.add_patch(inner)

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0)
plt.savefig('circles.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()