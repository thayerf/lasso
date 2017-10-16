env = Environment(
    CPPPATH = ['src'],
    CCFLAGS='-std=c++11 -g',
    LIBS=['armadillo','lapack','blas'],
    LINKFLAGS=['-g'])

# Doubles compilation time.
#env.Append(CCFLAGS='-O3 -msse2')

env.Program(
    target='test',
    source=[Glob('src/*.cpp')])
