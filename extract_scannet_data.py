
import os, sys, struct, argparse
import numpy as np
import zlib
import imageio
import cv2
import png

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  # def load(self, file_handle):
  #   self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
  #   self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
  #   self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
  #   self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
  #   self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
  #   self.color_data = ''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
  #   self.depth_data = ''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))
  def load(self, file_handle):
    self.camera_to_world = np.frombuffer(file_handle.read(16*4), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    # 直接读取字节数据
    self.color_data = file_handle.read(self.color_size_bytes)
    self.depth_data = file_handle.read(self.depth_size_bytes)

  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)

  def load(self, filename):
    with open(filename, 'rb') as f:
        version = struct.unpack('I', f.read(4))[0]
        assert self.version == version
        strlen = struct.unpack('Q', f.read(8))[0]
        # 将每个字节对象转换为对应的 ASCII 字符，并连接成字符串
        self.sensor_name = ''.join(chr(byte[0]) for byte in struct.unpack('c'*strlen, f.read(strlen)))
        
        # 对于浮点数，直接使用 numpy 的 frombuffer 函数来读取和转换数据
        self.intrinsic_color = np.frombuffer(f.read(16*4), dtype=np.float32).reshape(4, 4)
        self.extrinsic_color = np.frombuffer(f.read(16*4), dtype=np.float32).reshape(4, 4)
        self.intrinsic_depth = np.frombuffer(f.read(16*4), dtype=np.float32).reshape(4, 4)
        self.extrinsic_depth = np.frombuffer(f.read(16*4), dtype=np.float32).reshape(4, 4)
        
        # 对于整数，使用 struct.unpack 读取，然后根据索引获取值
        self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
        self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
        self.color_width = struct.unpack('I', f.read(4))[0]
        self.color_height = struct.unpack('I', f.read(4))[0]
        self.depth_width = struct.unpack('I', f.read(4))[0]
        self.depth_height = struct.unpack('I', f.read(4))[0]
        self.depth_shift = struct.unpack('f', f.read(4))[0]
        
        # 对于长整型
        num_frames = struct.unpack('Q', f.read(8))[0]
        self.frames = []
        for i in range(num_frames):
            frame = RGBDFrame()
            frame.load(f)
            self.frames.append(frame)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    # print 'exporting', len(self.frames)//frame_skip, ' depth frames to', output_path
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      #imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
      with open(os.path.join(output_path, str(f) + '.png'), 'wb') as f: # write 16-bit
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        depth = depth.reshape(-1, depth.shape[1]).tolist()
        writer.write(f, depth)

  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    # print 'exporting', len(self.frames)//frame_skip, 'color frames to', output_path
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    # print 'exporting', len(self.frames)//frame_skip, 'camera poses to', output_path
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    # print 'exporting camera intrinsics to', output_path
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))



# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--seq_name', required=True)
parser.add_argument('--data_path', default='/scratch/zdeng/datasets/scans', 
  help='path to sens file to read')
parser.add_argument('--output_path', default='/scratch/zdeng/datasets/scannet', 
  help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)

opt = parser.parse_args()
print(opt)


def main():
  sens_path = os.path.join(opt.data_path, opt.seq_name, f"{opt.seq_name}.sens")
  out_path = os.path.join(opt.output_path, opt.seq_name)
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  # load the data
  sys.stdout.write('loading %s...' % sens_path)
  sd = SensorData(sens_path)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(out_path, 'depth'))
  if opt.export_color_images:
    sd.export_color_images(os.path.join(out_path, 'color'))
  if opt.export_poses:
    sd.export_poses(os.path.join(out_path, 'pose'))
  if opt.export_intrinsics:
    sd.export_intrinsics(os.path.join(out_path, 'intrinsic'))


if __name__ == '__main__':
    main()