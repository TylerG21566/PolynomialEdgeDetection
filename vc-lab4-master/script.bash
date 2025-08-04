# sed -i 's/\r//' script.bash
cd /home/f78947tg/Desktop/VC/comp27112_lab4_code/
g++ horizon.cpp -o display `pkg-config --cflags --libs opencv4`
./display optic_nerve_head.jpg
