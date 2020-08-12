g++ -O3 -flto -g3 -ffast-math -fno-math-errno -march=native -fopenmp -mtune=native rast.cc -o rast -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic -lsdl2
