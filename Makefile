OPENCVFLAGS=`pkg-config --cflags --libs opencv4`

all: rl_deconv

rl_deconv: rl_deconv.cpp
	g++ -g rl_deconv.cpp -o rl_deconv $(OPENCVFLAGS)

clean:
	rm -f rl_deconv *.o