import moderngl
import torch
import os
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import pycuda.driver as cuda
    import pycuda.gl as cuda_gl


class GlHairRenderer:
    def __init__(
        self,
        hair_strands: torch.Tensor,
        head_verts: torch.Tensor = None,
        head_faces: torch.Tensor = None,
        image_size: tuple =(512, 512),
    ):
        self.device = hair_strands.device
        hair_strands = hair_strands.detach().cpu().numpy()
        
        ctx = moderngl.create_standalone_context(device_index=0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.gc_mode = "auto"
        
        head_fbo = ctx.simple_framebuffer(image_size, dtype="f4")
        head_fbo.use()
        head_fbo.clear()
        
        # head
        if head_verts is not None and head_faces is not None:
            head_verts = head_verts.detach().cpu().numpy()
            head_faces = head_faces.detach().cpu().numpy()
            ctx.simple_vertex_array(
                ctx.program(
                    vertex_shader="""
                        #version 330
                        in vec3 in_vert;
                        void main() {
                            float x = -in_vert.x;
                            float y = -in_vert.y;
                            float z = in_vert.z * 0.1;
                            gl_Position = vec4(x, y, z, 1.0);
                        }
                    """,
                    fragment_shader="""
                        #version 330
                        out vec4 f_color;
                        void main() {
                            f_color = vec4(0.0, 0.0, 0.0, 0.0);
                        }
                    """,
                ),
                ctx.buffer(head_verts.astype(np.float32)),
                "in_vert",
                index_buffer=ctx.buffer(head_faces.astype(np.int32)),
            ).render()
        
        # strands
        N, L, _ = hair_strands.shape
        index = np.arange(0, N * L, dtype=np.int32).reshape(N, L)
        index = np.concatenate([
            index, index[:, -1:], -np.ones((N, 1), dtype=np.int32)
        ], axis=1)
        hair_vbo = ctx.buffer(hair_strands.astype(np.float32))
        
        hair_vao = ctx.simple_vertex_array(
            ctx.program(
                vertex_shader="""
                    #version 330
                    in vec3 in_vert;
                    out int vertexIndex;
                    void main() {
                        float x = -in_vert.x;
                        float y = -in_vert.y;
                        float z = in_vert.z * 0.1;
                        gl_Position = vec4(x, y, z, 1.0);
                        vertexIndex = gl_VertexID;
                    }
                """,
                geometry_shader='''
                    #version 330
                    layout(lines_adjacency) in;
                    layout(line_strip, max_vertices = 3) out;

                    in  int vertexIndex[];
                    out vec3 color;

                    void main() {
                        vec2 tangent = normalize(gl_in[1].gl_Position.xy - gl_in[0].gl_Position.xy);
                        tangent.y = -tangent.y;
                        color = vec3(tangent * 0.5 + 0.5, gl_in[0].gl_Position.z);
                        gl_Position = gl_in[0].gl_Position;
                        EmitVertex();
                        
                        tangent = normalize(gl_in[2].gl_Position.xy - gl_in[1].gl_Position.xy);
                        tangent.y = -tangent.y;
                        color = vec3(tangent * 0.5 + 0.5, gl_in[1].gl_Position.z);
                        gl_Position = gl_in[1].gl_Position;
                        EmitVertex();
                        
                        if (vertexIndex[2] == vertexIndex[3]) {
                            color = vec3(tangent * 0.5 + 0.5, gl_in[2].gl_Position.z);
                            gl_Position = gl_in[2].gl_Position;
                            EmitVertex();
                        }

                        EndPrimitive();
                    }
                ''',
                fragment_shader="""
                    #version 330
                    in vec3 color;
                    out vec4 f_color;
                    void main() {
                        f_color = vec4(color, 1.0);
                    }
                """,
            ),
            hair_vbo,
            "in_vert",
            index_buffer=ctx.buffer(index),
        )
        
        texture = ctx.texture(image_size, 4, dtype="f4")
        tex_buffer = ctx.buffer(reserve=image_size[0] * image_size[1] * 4 * 4)
        depth_buffer = ctx.depth_renderbuffer(image_size)
        hair_fbo = ctx.framebuffer(color_attachments=[texture], depth_attachment=depth_buffer)
        hair_fbo.use()
        
        # attributes
        self.ctx = ctx
        self.image_size = image_size
        self.texture = texture
        self.tex_buffer = tex_buffer
        self.head_fbo = head_fbo
        self.hair_fbo = hair_fbo
        self.hair_vao = hair_vao
        self.hair_vbo = hair_vbo
        
        # cuda context
        cuda.init()
        device = cuda.Device(self.device.index)
        cuda_ctx = device.make_context()
        hair_cuda_buffer = cuda_gl.RegisteredBuffer(hair_vbo.glo)
        tex_cuda_buffer = cuda_gl.RegisteredBuffer(tex_buffer.glo)
        cuda_ctx.pop()
        
        self.cuda_ctx = cuda_ctx
        self.hair_cuda_buffer = hair_cuda_buffer
        self.tex_cuda_buffer = tex_cuda_buffer
        self.result_tensor = torch.zeros((image_size[0], image_size[1], 4),
                                  dtype=torch.float32, device=self.device).contiguous()
        
    def __del__(self):
        torch.cuda.synchronize()
        self.cuda_ctx.synchronize()
        self.ctx.finish()
        
        self.cuda_ctx.push()
        self.hair_cuda_buffer.unregister()
        self.tex_cuda_buffer.unregister()
        self.cuda_ctx.pop()
        
        self.ctx.release()
    
    def _tensor_to_opengl(
        self, tensor: torch.Tensor, cuda_buffer: cuda_gl.RegisteredBuffer
    ):
        self.cuda_ctx.push()
        mapping_obj = cuda_buffer.map()
        data_ptr, sz = mapping_obj.device_ptr_and_size()
        cuda.memcpy_dtod(data_ptr, tensor.data_ptr(), sz)
        mapping_obj.unmap()
        self.cuda_ctx.pop()
    
    def _opengl_to_tensor(
        self, tensor: torch.Tensor, cuda_buffer: cuda_gl.RegisteredBuffer
    ):
        self.cuda_ctx.push()
        mapping_obj = cuda_buffer.map()
        data_ptr, sz = mapping_obj.device_ptr_and_size()
        cuda.memcpy_dtod(tensor.data_ptr(), data_ptr, sz)
        mapping_obj.unmap()
        self.cuda_ctx.pop()
    
    def render(self, hair_strands: torch.Tensor = None):
        self.ctx.copy_framebuffer(self.hair_fbo, self.head_fbo)
        
        # write vertex buffer
        if hair_strands is not None:
            self.device = hair_strands.device
            hair_strands = hair_strands.contiguous()
            torch.cuda.synchronize()
            self._tensor_to_opengl(hair_strands, self.hair_cuda_buffer)
            self.cuda_ctx.synchronize()

        # render
        self.hair_vao.render(moderngl.LINE_STRIP_ADJACENCY)
        
        # read result
        image = self.result_tensor
        self.texture.read_into(self.tex_buffer)
        self._opengl_to_tensor(image, self.tex_cuda_buffer)
        
        # image post process
        image = image.to(self.device)
        img_silh = image[:, :, 3].clip(0, 1)
        img_depth = image[:, :, 2]
        z_min = img_depth[img_depth > 0].min()
        img_depth = 1 - (img_depth - z_min) / (img_depth.max() - z_min)
        img_depth = (img_depth * img_silh).clip(0, 1)
        img_orien = torch.cat((image[:, :, :2], img_silh[..., None]*0.5), axis=2).clip(0, 1)
        
        return img_orien, img_silh, img_depth


if __name__ == "__main__":
    import time
    import glob
    import pyexr
    import matplotlib.pyplot as plt
    
    from pytorch3d.io import load_hair
    from .utils import load_obj_with_uv
    from .render_utils import load_cameras, transform_points_to_ndc
    from .hair_utils import resample_strands_fast
    
    device = torch.device("cuda")
    
    hair_name = "DD_0528_02_changfa_hair002"
    head_path = os.path.join("X:/contrastive_learning/data/assets/scalp_models",
                                        hair_name[:10]+".obj")
    hair_path = glob.glob(os.path.join("X:/contrastive_learning/data/assets", "*/*",
                                        hair_name+"_Wo_Modifiers_resample_32.hair"))[0]
    camera_path = os.path.join(os.path.dirname(hair_path), "camera.json")

    cameras = load_cameras(camera_path, device=device)
    head_mesh = load_obj_with_uv(head_path, device=device)
    head_verts_ndc = transform_points_to_ndc(cameras, head_mesh.verts_packed())
    hair_strands = resample_strands_fast(load_hair(hair_path), 32).to(device)
    gl_render = GlHairRenderer(
        hair_strands, head_verts_ndc, head_mesh.faces_packed())

    hair_strands_proj = transform_points_to_ndc(cameras, hair_strands)
    start = time.time()
    img_orien, img_silh, img_depth = gl_render.render(hair_strands_proj)
    end = time.time()
    print("full render time:", end - start)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_orien.cpu())
    axes[1].imshow(img_silh.cpu())
    axes[2].imshow(img_depth.cpu())
    plt.show()
    
    data = {'default': img_orien.cpu().numpy(), 'orientation': img_orien.cpu().numpy(), 'depth': img_depth.cpu().numpy()}
    pyexr.write("output/opengl/result.exr", data)
