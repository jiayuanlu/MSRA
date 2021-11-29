# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(name='mylinear_cpp',
#       ext_modules=[cpp_extension.CppExtension('mylinear_cpp', ['mylinear.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
setup(
      name='mylinear_cpp',
      ext_modules=[
      CppExtension(
            name='mylinear_cpp',
            sources=['mylinear.cpp'],
            extra_compile_args=['-g']),
      ],
      cmdclass={
      'build_ext': BuildExtension
      })