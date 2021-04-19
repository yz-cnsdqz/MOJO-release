import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx
import cv2
import pickle
import pdb
sys.path.append(os.getcwd())
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from experiments.utils.vislib import *




def visualize(data, outfile_path=None, datatype='gt'):
    ## prepare data
    n_frames = data.shape[0]
    data = data.reshape(n_frames, 22, 3)
    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=50
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()


    ## create body mesh from data
    ball_list = []
    for i in range(22):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)

    ## create line on limbs
    limb_lines = o3d.geometry.LineSet()
    vis.add_geometry(limb_lines)
    vis.poll_events()
    vis.update_renderer()


    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            b.translate(data[it,i], relative=False)
            vis.update_geometry(b)

        ## plot limbs
        limb_lines.points = o3d.utility.Vector3dVector(data[it, [1,4,7,2,5,8,16,18,20,17,19,21]])
        limb_lines.colors = o3d.utility.Vector3dVector(np.zeros([8,3]))
        lines = []
        lines.append(np.array([0, 1]))
        lines.append(np.array([1, 2]))
        lines.append(np.array([3, 4]))
        lines.append(np.array([4, 5]))
        lines.append(np.array([6, 7]))
        lines.append(np.array([7, 8]))
        lines.append(np.array([9, 10]))
        lines.append(np.array([10, 11]))
        limb_lines.lines = o3d.utility.Vector2iVector(np.stack(lines,axis=0))
        vis.update_geometry(limb_lines)

        if it <15:
            for ball in ball_list:
                ball.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
        else:
            if datatype == 'gt':
                pass
            else:
                for ball in ball_list:
                    ball.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"

        ## update camera. the camera follows the human body
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0)
        body_t = np.array([0,0,0])
        cam_t = body_t + 3.5*np.ones(3)
        ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        # if it == 35:
        #     o3d.io.write_line_set('tmp_seq20_gen0_lineset_frame35.ply', limb_lines)
        #     for i, b in enumerate(ball_list):
        #         o3d.io.write_triangle_mesh('tmp_seq20_gen0_frame35_{}.ply'.format(i), b)

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:03d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)


if __name__=='__main__':
    results_file_name = sys.argv[1]
    data = np.load(results_file_name)
    n_seq=data.shape[0]
    n_gen=data.shape[1]
    for seq in range(n_seq):
        for gen in range(n_gen):
            renderfolder = results_file_name+'_renderjts/seq{}_gen{}'.format(seq, gen)
            if not os.path.exists(renderfolder):
                os.makedirs(renderfolder)
            visualize(data[seq, gen], renderfolder, datatype='dlow')