CC = g++
CFLAG = -Wall -w -O3 -msse2 -ggdb -fPIC

SRC = ../src
OBJ = ../obj
LIB = ../lib
BIN = ../bin

LIB_FLAG = -L../lib -L/usr/lib64 -L/usr/local/lib64 -L/usr/local/lib
#INCLUDE = -I. -I../include -I/usr/include -I/usr/sys/include -I/usr/local/include/opencv2.1
#MY_LIB = -lpthread -lcv -lcvaux -lhighgui -lml -lcxcore -lm 

INCLUDE = -I. -I../include -I/usr/include -I/usr/sys/include -I/usr/local/include/opencv -I/usr/local/include/opencv2 -I/usr/local/include
MY_LIB = -lpthread -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml

#LibObjs = $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(wildcard $(SRC)/*.cpp))
LibObjs = 	$(OBJ)/ImageProcessing.o \
			$(OBJ)/ColorLayout.o \
			$(OBJ)/EdgeHist.o \
			$(OBJ)/API_imagequality.o \
			$(OBJ)/SD_global.o 
DemoObjs = $(OBJ)/Demo.o

all: Demo
Demo: $(BIN) $(LIB) $(OBJ) $(DemoObjs) lib
	@echo ---------------------------------------------------------------------------------------------------------
	$(CC) $(CFLAG) $(INCLUDE) -o $(BIN)/Demo $(DemoObjs) -lIN_IMAGE_SIMILAR_DETECT_1_0_0 $(LIB_FLAG) $(MY_LIB)
	@echo ---------------------------------------------------------------------------------------------------------

$(BIN) $(LIB) $(OBJ):
	@mkdir -p $@

$(DemoObjs): %.o:
	@echo ---------------------------------------------------------------------------------------------------------
	$(CC) -c $(CFLAG) $(INCLUDE) -o $@ $(SRC)/$(@F:.o=.cpp)    
	@echo ---------------------------------------------------------------------------------------------------------

lib: $(LIB) $(OBJ) $(LibObjs)
	@echo ---------------------------------------------------------------------------------------------------------
	ar -rv $(LIB)/libIN_IMAGE_SIMILAR_DETECT_1_0_0.a $(LibObjs)
	@echo ---------------------------------------------------------------------------------------------------------
	
$(LibObjs): %.o:
	@echo ---------------------------------------------------------------------------------------------------------
	$(CC) -c $(CFLAG) $(INCLUDE) -o $@ $(SRC)/$(@F:.o=.cpp)    
	@echo ---------------------------------------------------------------------------------------------------------

.PHONY : clean
clean :
	rm -f $(BIN)/Demo $(LIB)/libIN_IMAGE_SIMILAR_DETECT_1_0_0.a core $(DemoObjs) $(LibObjs)
