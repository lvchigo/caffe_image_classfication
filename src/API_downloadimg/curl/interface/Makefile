CXX := g++


LOG_DIR = ../../../io/log
LOG_SRC =
COM_DIR = ../../../common

CXXFLAGS := -g \
			-I$(LOG_DIR) -I$(COM_DIR)

CXXLIBS  := -lcurl -lpthread

HEADERS := downloader.h

all:demo mdemo mwget

demo: demo.o downloader.o
	$(CXX) $(CXXFLAGS) -o demo demo.o downloader.o $(CXXLIBS)

mdemo: mdemo.cpp downloader.o
	$(CXX) $(CXXFLAGS) mdemo.cpp -o mdemo  $(CXXLIBS)


downloader.o: downloader.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c downloader.cpp

mwget:mwget.cpp
	$(CXX) $(CXXFLAGS) mwget.cpp -o mwget $(CXXLIBS)

demo.o: demo.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c demo.cpp 

clean:
	rm -f *.o
	rm -f demo

