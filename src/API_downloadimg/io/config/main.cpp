
#include <iostream>

#include "config.h"

int main(int argc, char* argv[])
{
	std::string conf_file("main.config");
	char** envp = 0;
	Config config(conf_file, envp);

	std::cout<<"server host = "<<config.pString("server.host")<<std::endl;
	std::cout<<"server port = "<<config.pInt("server.port")<<std::endl;

	std::cout<<"redis host = "<<config.pString("redis.host")<<std::endl;
	std::cout<<"redis port = "<<config.pInt("redis.port")<<std::endl;
	std::cout<<"redis db = "<<config.pInt("redis.db")<<std::endl;

	std::cout<<"images dir: "<<config.pString("dir.images")<<std::endl;
	std::cout<<"features dir: "<<config.pString("dir.features")<<std::endl;
	return 0;
	
}
