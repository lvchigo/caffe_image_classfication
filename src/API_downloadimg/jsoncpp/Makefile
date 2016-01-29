
CXX = g++


JC_INC = -I./include
JC_SRC = src/lib_json/json_reader.cpp  src/lib_json/json_value.cpp  src/lib_json/json_writer.cpp

CXXFLAGS = $(JC_INC)
LDFLAGS =

OBJS = $(JC_SRC:%.cpp=%.o)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: demo

demo:$(OBJS) demo.cpp
	@echo $(OBJS)
	$(CXX)  $(CXXFLAGS) demo.cpp  -o demo   $(OBJS) $(LDFLAGS)

clean:
	rm -f demo
	rm -f $(OBJS)
