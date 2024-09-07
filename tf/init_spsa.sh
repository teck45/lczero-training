
python3 -m venv spsaenv
source spsaenv/bin/activate
python3 -m pip install numpy protobuf
git clone https://github.com/Ergodice/lc0.git -b update lc0-src
cd lc0-src
./build.sh
# cd ..
# mkdir syzygy
# cd syzygy
# wget --mirror --no-parent --no-directories -e robots=off http://tablebase.sesse.net/syzygy/3-4-5/
cd ..
mkdir lc0
cp lc0-src/build/release/lc0 ./lc0
cd lc0
mkdir nets
cd nets
wget https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz
mv t1-256x10-distilled-swa-2432500.pb.gz t1d.pb.gz
cp t1d.pb.gz t1d-0.pb.gz 
cd ..
cd ..
mkdir opbooks
cd opbooks
rm *
wget https://storage.lczero.org/files/uho.pgn
# apt install unzip
# # unzip UHO_Lichess_4852_v1.epd.zip
# rm *.zip
cd ..
git clone https://github.com/Ergodice/lczero-training.git -b spsa