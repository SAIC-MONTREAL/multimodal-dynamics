import os
import numpy as np

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pose_tensorboard(pose_1=np.array([[0, 0, 0, 1, 0, 0, 0]]), pose_2=np.array([[0, 0, 0, 1, 0, 0, 0]]),
                          axis_lim=2, normalized_quaternions=True, show=False, seq_length=30):
    pose_1 = np.array(pose_1)
    pose_2 = np.array(pose_2)

    positions_1, quaternions_1 = pose_1[:, :3], pose_1[:, 3:]
    positions_2, quaternions_2 = pose_2[:, :3], pose_2[:, 3:]

    if normalized_quaternions:
        quaternions_1 = 2 * quaternions_1 - 1
        quaternions_2 = 2 * quaternions_2 - 1

    n_rows = len(positions_1) // seq_length
    fig = plt.figure(figsize=(3 * seq_length, 3 * n_rows))

    for i, (position_1, quaternion_1, position_2, quaternion_2) in enumerate(zip(positions_1, quaternions_1, positions_2, quaternions_2)):
        ax = fig.add_subplot(n_rows, seq_length, i + 1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim((-axis_lim, axis_lim))
        ax.set_ylim((-axis_lim, axis_lim))
        ax.set_zlim((-axis_lim, axis_lim))

        colors = ['r', 'g', 'b']
        lines_1 = sum([ax.plot([], [], [], c=c, linestyle='-', linewidth=2) for c in colors], [])
        lines_2 = sum([ax.plot([], [], [], c=c, linestyle='--', linewidth=2) for c in colors], [])

        for line in lines_1 + lines_2:
            line.set_data([], [])
            line.set_3d_properties([])

        plot_lines(position_1, quaternion_1, lines_1)
        plot_lines(position_2, quaternion_2, lines_2)

        fig.canvas.draw()
    if show:
        plt.show()

    return fig


def plot_single_pose_tensorboard(pose=np.array([[0, 0, 0, 1, 0, 0, 0]]), axis_lim=2, normalized_quaternions=True, show=False,
                                 seq_length=20):
    pose = np.array(pose)

    positions, quaternions = pose[:, :3], pose[:, 3:]

    if normalized_quaternions:
        quaternions = 2 * quaternions - 1

    n_rows = len(positions) // seq_length
    fig = plt.figure(figsize=(3 * seq_length, 3 * n_rows))

    for i, (position, quaternion) in enumerate(zip(positions, quaternions)):
        ax = fig.add_subplot(n_rows, seq_length, i + 1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim((-axis_lim, axis_lim))
        ax.set_ylim((-axis_lim, axis_lim))
        ax.set_zlim((-axis_lim, axis_lim))

        colors = ['r', 'g', 'b']
        lines = sum([ax.plot([], [], [], c=c, linestyle='-', linewidth=2) for c in colors], [])

        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

        plot_lines(position, quaternion, lines)

        fig.canvas.draw()
        if show:
            plt.show()

    return fig


def plot_lines(position, quaternion, lines):
    startpoints = np.array([position, position, position])
    endpoints = np.array([position + np.array([2, 0, 0]), position + np.array([0, 2, 0]), position + np.array([0, 0, 2])])

    q = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

    for line, start, end in zip(lines, startpoints, endpoints):
        start = q.rotate(start)
        end = q.rotate(end)

        line.set_data([start[0], end[0]], [start[1], end[1]])
        line.set_3d_properties([start[2], end[2]])


def plot_pose(output, target, plot_dir, title, show=False, seq_length=30, axis_lim=2,
              normalized_quaternions=True, sv=False):
    n_figs = len(output) // seq_length

    pose_1 = output.cpu().detach().numpy()
    pose_2 = target.cpu().detach().numpy()

    positions_1, quaternions_1 = pose_1[:, :3], pose_1[:, 3:]
    positions_2, quaternions_2 = pose_2[:, :3], pose_2[:, 3:]

    if normalized_quaternions:
        quaternions_1 = 2 * quaternions_1 - 1
        quaternions_2 = 2 * quaternions_2 - 1

    if not sv:
        n_figs = len(positions_1) // seq_length

        for i in range(n_figs):
            fig = plt.figure(figsize=(seq_length , 1))
            plt.subplots_adjust(top=0.98, right=0.98, left=0.02, bottom=0.1, wspace=0.1, hspace=0.01)

            for j in range(seq_length):
                ax = fig.add_subplot(1, seq_length, j + 1, projection='3d')
                # ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
                # ax.set_xticks([-2, 0, 2]), ax.set_yticks([-2, 0, 2]), ax.set_zticks([-2, 0, 2])
                ax.set_xlim((-axis_lim, axis_lim))
                ax.set_ylim((-axis_lim, axis_lim))
                ax.set_zlim((-axis_lim, axis_lim))
                ax.tick_params(axis='both', labelbottom=False, labelleft=False, labelright=False, labeltop=False)

                colors = ['r', 'g', 'b']
                lines_1 = sum([ax.plot([], [], [], c=c, linestyle='-', linewidth=2) for c in colors], [])
                lines_2 = sum([ax.plot([], [], [], c=c, linestyle='--', linewidth=2) for c in colors], [])

                for line in lines_1 + lines_2:
                    line.set_data([], [])
                    line.set_3d_properties([])

                idx = (i - 1) * seq_length + j
                plot_lines(positions_1[idx, :], quaternions_1[idx, :], lines_1)
                plot_lines(positions_2[idx, :], quaternions_2[idx, :], lines_2)

                fig.canvas.draw()

            if show:
                plt.show()

            fig.savefig(os.path.join(plot_dir, title + "_" + str(i)), dpi=300)
            plt.close(fig)
    else:
        n_rows = len(positions_1) // seq_length
        fig = plt.figure(figsize=(seq_length/2, seq_length/2))
        plt.subplots_adjust(top=0.98, right=0.98, left=0.02, bottom=0.02, wspace=0.1, hspace=0.1)

        for i in range(n_rows):
            for j in range(seq_length):
                ax = fig.add_subplot(n_rows, seq_length, i * seq_length + j + 1, projection='3d')
                # ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
                # ax.set_xticks([-2, 0, 2]), ax.set_yticks([-2, 0, 2]), ax.set_zticks([-2, 0, 2])
                ax.tick_params(axis='both', labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax.set_xlim((-axis_lim, axis_lim))
                ax.set_ylim((-axis_lim, axis_lim))
                ax.set_zlim((-axis_lim, axis_lim))

                colors = ['r', 'g', 'b']
                lines_1 = sum([ax.plot([], [], [], c=c, linestyle='-', linewidth=2) for c in colors], [])
                lines_2 = sum([ax.plot([], [], [], c=c, linestyle='--', linewidth=2) for c in colors], [])

                for line in lines_1 + lines_2:
                    line.set_data([], [])
                    line.set_3d_properties([])

                idx = (i - 1) * seq_length + j
                plot_lines(positions_1[idx, :], quaternions_1[idx, :], lines_1)
                plot_lines(positions_2[idx, :], quaternions_2[idx, :], lines_2)

                fig.canvas.draw()

            if show:
                plt.show()

        fig.savefig(os.path.join(plot_dir, title + "_" ), dpi=300)
        plt.close(fig)
