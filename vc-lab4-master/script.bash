# sed -i 's/\r//' script.bash
cd <folder_directory>
g++ horizon.cpp -o display `pkg-config --cflags --libs opencv4`
./display optic_nerve_head.jpg
