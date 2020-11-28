import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import smplx
import cv2
import pickle
import pdb

sys.path.append(os.getcwd())
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from experiments.utils.vislib import *


import json
with open('/home/yzhang/body_models/Mosh_related/CMU.json') as f:
                markerdict = json.load(f)['markersets'][0]['indices']

markers = list(markerdict.values())
head_markers = [markers.index(markerdict['RFHD']),
                    markers.index(markerdict['LFHD']),
                    markers.index(markerdict['RBHD']),
                    markers.index(markerdict['LBHD'])]
uppertorso_markers = [markers.index(markerdict['RSHO']),
                            markers.index(markerdict['LSHO']),
                            markers.index(markerdict['CLAV']),
                            markers.index(markerdict['C7'])]
lowertorso_markers = [markers.index(markerdict['RFWT']),
                            markers.index(markerdict['LFWT']),
                            markers.index(markerdict['LBWT']),
                            markers.index(markerdict['RBWT'])]


def visualize(gender, betas, transf_rotmat, transf_transl,
                data, outfile_path=None, datatype='gt',
                seq=0, gen=0):
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

    # create a virtual environment
    ### ground
    x_range = np.arange(-200, 200, 0.75)
    y_range = np.arange(-200, 200, 0.75)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    vis.add_geometry(gp_lines)
    vis.poll_events()
    vis.update_renderer()
    vis.add_geometry(gp_pcd)
    vis.poll_events()
    vis.update_renderer()

    ### top lighting
    box = o3d.geometry.TriangleMesh.create_box(width=200, depth=1,height=200)
    box.translate(np.array([-200,-200,6]))
    vis.add_geometry(box)
    vis.poll_events()
    vis.update_renderer()

    ### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh in open3d
    body = o3d.geometry.TriangleMesh()
    vis.add_geometry(body)
    vis.poll_events()
    vis.update_renderer()

    pcd = o3d.geometry.PointCloud()


    ## parse smplx parameters
    bm = get_body_model('smplx', gender, 60, device='cpu')
    bparam = {}
    bparam['transl'] = data[:,:3]
    bparam['global_orient'] = data[:,3:6]
    bparam['betas'] = np.tile(betas[None,...], (60,1))
    bparam['body_pose'] = data[:,6:69]
    bparam['left_hand_pose'] = data[:,69:81]
    bparam['right_hand_pose'] = data[:,81:]

    ## from amass coord to world coord
    global_ori_a = R.from_rotvec(bparam['global_orient']).as_dcm() # to [t,3,3] rotation mat
    global_ori_w = np.einsum('ij,tjk->tik', transf_rotmat, global_ori_a)
    bparam['global_orient'] = R.from_dcm(global_ori_w).as_rotvec()
    bparam['transl'] = np.einsum('ij,tj->ti', transf_rotmat, bparam['transl']) + transf_transl

    ## obtain body mesh sequences
    for key in bparam:
        bparam[key] = torch.FloatTensor(bparam[key])
    verts_seq = bm(return_verts=True, **bparam).vertices.detach().cpu().numpy() #[t,verts, 3]
    kps_seq = verts_seq[:,markers,:]

    ## main loop for rendering
    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        ## set body mesh locations
        body.vertices = o3d.utility.Vector3dVector(verts_seq[it])
        body.triangles = o3d.utility.Vector3iVector(bm.faces)
        body.vertex_normals = o3d.utility.Vector3dVector([])
        body.triangle_normals = o3d.utility.Vector3dVector([])
        body.compute_vertex_normals()
        vis.update_geometry(body)

        ## set body mesh color
        if it <15:
            body.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
        else:
            body.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"

        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(15)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let the cam follow the body
        body_t = np.array([0,0,0])
        if 'ACCAD' in results_file_name:
            cam_t = body_t + 3.5*np.array([1,1,1])
        elif 'BMLhandball' in results_file_name:
            cam_t = body_t + 3.0*np.array([-1,-1,1.2])
        else:
            cam_t = body_t + 3.5*np.array([1,1,1])
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

        ## want to save intermediate results? Then uncomment these
        # if it in [0,14,20,25,30,35, 40,45, 50,59]:
        #     # o3d.visualization.draw_geometries([limb_lines]+ball_list)
        #     o3d.io.write_triangle_mesh('tmp_seq{}_gen{}_body_frame{}.ply'.format(seq,gen,it), body)
        #     o3d.io.write_point_cloud("tmp_seq{}_gen{}_kps_frame{}.ply".format(seq,gen,it), pcd)

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:03d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(10)


if __name__=='__main__':

    proj_path = os.getcwd()
    exps = [
            'amass_vanilla_nsamp50',
            'amass_mojo_f9_nsamp50'
            ]
    datasets = ['ACCAD', 'BMLhandball']
    for exp in exps:
        for data in datasets:
            results_file_name = proj_path+'/results/{}/results/seq_gen_seed1_optim/{}/results_marker_41.pkl'.format(exp, data)
            print('-- processing: '+results_file_name)

            with open(results_file_name, 'rb') as f:
                data = pickle.load(f)
            algos = ['dlow']
            for algo in algos:
                dd = data[algo+'_smplx_params']
                n_seq=dd.shape[0]
                n_gen=dd.shape[1]
                for seq in range(n_seq):
                    gender = data['gender'][seq]
                    betas = data['betas'][seq]
                    transf_rotmat = data['transf_rotmat'][seq]
                    transf_transl = data['transf_transl'][seq]
                    for gen in range(n_gen):
                        renderfolder = results_file_name+'_renderbm_{}_seq{}_gen{}'.format(algo,seq, gen)
                        if not os.path.exists(renderfolder):
                            os.makedirs(renderfolder)
                        else:
                            pass
                        visualize(gender, betas, transf_rotmat, transf_transl,
                                    dd[seq, gen], renderfolder, datatype=algo,
                                    seq=seq, gen=gen)