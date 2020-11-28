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




def visualize(data, outfile_path=None, datatype='gt',seq=0,gen=0):
    ## prepare data
    n_frames = data.shape[0]

    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=10
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()

    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh from data
    ball_list = []
    for i in range(41):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)

    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            b.translate(data[it,i], relative=False)
            vis.update_geometry(b)

        if it <10:
            for ball in ball_list:
                ball.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
        else:
            if datatype == 'gt':
                pass
            else:
                for ball in ball_list:
                    ball.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"

        # ## special colors on head (for debug)
        # ball_list[22].paint_uniform_color(color_hex2rgb('#5874ae'))
        # ball_list[7].paint_uniform_color(color_hex2rgb('#9ebf5e'))
        # ball_list[3].paint_uniform_color(color_hex2rgb('#b25d48'))
        # ball_list[18].paint_uniform_color(color_hex2rgb('#749a83'))

        # if it in [0,15,30,45,59]:
        #     # o3d.visualization.draw_geometries([limb_lines]+ball_list)
        #     # o3d.io.write_line_set('tmp_seq20_gen0_lineset_frame35.ply', limb_lines)
        #     for i, b in enumerate(ball_list):
        #         o3d.io.write_triangle_mesh('tmp_seq{}_gen{}_kps{}_frame{}.ply'.format(seq,gen,i,it), b)

        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        cam_t = body_t + 2.0*np.ones(3)
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

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:03d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)


if __name__=='__main__':
    proj_path = os.getcwd()
    exps = [
            'amass_vanilla_nsamp50',
            'amass_mojo_f9_nsamp50'
            ]
    for exp in exps:
        results_file_name = proj_path+'/results/{}/results/seq_gen_seed1_optim/ACCAD/results_marker_41.pkl'.format(exp)
        print('-- processing: '+results_file_name)
        with open(results_file_name, 'rb') as f:
            data = pickle.load(f)
        algos = ['dlow']
        for algo in algos:
            dd = data[algo]
            n_seq=dd.shape[0]
            n_gen=dd.shape[1]
            for seq in range(n_seq):
                for gen in range(n_gen):
                    renderfolder = results_file_name+'_renderkps_{}_seq{}_gen{}'.format(algo,seq, gen)
                    if not os.path.exists(renderfolder):
                        os.makedirs(renderfolder)
                    visualize(dd[seq, gen], renderfolder, datatype=algo,
                            seq=seq,gen=gen)