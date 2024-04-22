# %%
import matplotlib.pyplot as plt

from matplotlib import ticker

import matplotlib as mpl


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


# plt.subplots_adjust(bottom=0.2, left=0.2)


# Define the step values and positions, including placeholders for the symbolic parts
P_values = [1, 1, 2, 3] + ['...'] + [3.5, 2.5, 1.5, 0.5]  # Placeholder values
k_values = list(range(len(P_values)))  # Placeholder positions
print(k_values)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(9, 5.0))

# Plot the steps before the ellipsis
ax.step(k_values[:4], P_values[:4], where='pre', color='black')

# Plot the steps after the ellipsis
ax.step(k_values[5:], P_values[5:], where='post', color='black')

# Adding the ellipsis symbol
ax.text(k_values[4], P_values[3], '. . .', ha='center', va='center', fontsize=20)

# Drawing vertical lines at P1 and P8
ax.vlines(x=0, ymin=0, ymax=P_values[0], color='black')
ax.vlines(x=k_values[-1], ymin=0, ymax=P_values[-1], color='black')

# Setting labels
ax.set_xlabel('$k$', fontsize=18)
ax.set_ylabel('$P(k)$', fontsize=18)

# Annotating the step values at P1 with '0'
ax.text(0.5, P_values[1]+0.05, '$P_1$', ha='center', va='bottom', fontsize=18)
ax.text(1.5, P_values[2]+0.05, '$P_2$', ha='center', va='bottom', fontsize=18)
ax.text(2.5, P_values[3]+0.05, '$P_3$', ha='center', va='bottom', fontsize=18)
ax.text(5.5, P_values[5]+0.05, '$P_{s-2}$', ha='center', va='bottom', fontsize=18)
ax.text(6.5, P_values[6]+0.05, '$P_{s-1}$', ha='center', va='bottom', fontsize=18)
ax.text(7.5, P_values[7]+0.05, '$P_{s}$', ha='center', va='bottom', fontsize=18)

# Set the x-axis ticks to only the defined k values, including kmax and excluding P labels
ax.set_xticks([0, 8])
ax.set_xticklabels(['$0$', '$k_{\mathrm{max}}$'],  fontsize=18)

# Remove y-axis ticks
ax.yaxis.set_ticks([])
# ax.yaxis.set_ticklabels(fontsize=14)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# # make arrows
# ax.plot((1), (-0.10), ls="", marker=">", ms=5, color="k",
#         transform=ax.get_yaxis_transform(), clip_on=False)
# ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
#         transform=ax.get_xaxis_transform(), clip_on=False)
plt.subplots_adjust(bottom=0.2)

plt.savefig('power_spectrum_shape.png', dpi=200)
# Show the plot
plt.show()
# %%