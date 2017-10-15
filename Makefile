main.o:
	g++ -Wall -fexceptions -O3 -std=c++11 -Iinclude -c main.cpp -o obj/Release/main.o

clasificador.o:
	g++ -Wall -fexceptions -O3 -std=c++11 -Iinclude -c src/redneuronal.cpp -o obj/Release/src/clasificador.o

objects: main.o clasificador.o

all:
	g++ -Wall -o bin/Release/clasificador obj/Release/main.o obj/Release/src/clasificador.o  -s -Lsrc/ -llogfile -larmadillo -lopenblas

clean:
	rm obj/Release/*.o; rm obj/Release/src/*.o; rm bin/Release/*;
