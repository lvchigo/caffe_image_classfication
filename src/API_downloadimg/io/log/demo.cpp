
#include <iostream>

#include "log.h"

int main(int argc, char* argv[])
{
	Logger log;
	
	log.set_file("c.log");
	
	log.set_level(3);

	log.info()<<"i"<<std::endl;
	
	log.warning()<<"w"<<std::endl;

	log.error()<<"e"<<std::endl;

	log.fatal()<<"f"<<std::endl;
	
	return 0;
}
