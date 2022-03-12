import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import os

#python wave_to_midi.py 読み込みファイルパス 出力ファイルパス

#4分音符の分解能(delta_time)が480でテンポ(tempo)が150なら、1つの4分音符が60/150=0.4秒なので、
#1区切りあたり0.03秒にしたければdiv_delta=480*(0.03/0.4)=36

#定数
delta_time=480
tempo=150
start_wait=960 #61以上
div_delta=36  #36
noteset_min=45 #110Hzのラの番号
checking_freq_num=73
checking_freq_min=110.0
super_sample=0
one_sample=8000
binary_sec=65536

def MakeNotes():
  global one_sample
  #初期設定
  cutting_time=60*div_delta/(delta_time*tempo)
  checking_freq_array=np.array([])
  for i in range(checking_freq_num+1):
    checking_freq_array=np.append(checking_freq_array,checking_freq_min*2**((2*i-1)/24.0))
  crit_freq_array=np.array([])
  for i in range(checking_freq_num):
    crit_freq_array=np.append(crit_freq_array,checking_freq_min*2**(i/12.0))
  noteset_array=np.array([],dtype=np.int16)
  for i in range(checking_freq_num):
    noteset_array=np.append(noteset_array,noteset_min+i)
  
  #音声ファイル読み込み
  args=sys.argv
  wav_filename=args[1]
  rate,data=scipy.io.wavfile.read(wav_filename)
  
  #ここでフーリエ変換を区間のみで行うかを変える
  if super_sample==0:
    one_sample=int(rate*cutting_time)
  
  #（振幅）の配列を作成(ステレオならモノラルに変える)
  if data.ndim>=2:
    data = data / (32768*data.shape[1])
    prevdata=np.zeros(data.shape[0])
    for i in range(data.shape[1]):
      prevdata+=data[:,i]
  if data.ndim==1:
    prevdata=data/32768
  
  #ほんとは分割した1つのデータの要素数が2の累乗のほうが計算が高速化されるが、今回は速度重視ではないので適当な長さにする
  cutting_sample_num=int(rate*cutting_time)
  
  #計算しやすいようにデータ要素数をcutting_sample_num倍に補完
  prevdata=np.append(prevdata,np.zeros(cutting_sample_num-prevdata.shape[0]%cutting_sample_num))
  
  #データ分割してmidi配列を作っていく
  outwave=np.array([])
  samplerange=int(prevdata.shape[0]/cutting_sample_num)
  prevdata=np.append(prevdata,np.zeros(cutting_sample_num*int(1+one_sample/cutting_sample_num)))
  volumepile=np.array([])
  for i in range(samplerange):
    cutdata=prevdata[cutting_sample_num*i:cutting_sample_num*i+one_sample-1]
    output=np.fft.fftfreq(cutdata.shape[0],d=1.0/rate)
    output=np.vstack((output,np.abs(np.fft.fft(cutdata))))
    output=output.T
    output=output[np.argsort(output[:,0])]
    output=output.T
    volumedata=np.array([])
    for j in range(checking_freq_num):
      distancemax=np.max(output[1,:]*((output[0,:]>=checking_freq_array[j])&(output[0,:]<checking_freq_array[j+1])).astype(np.float64))
      volumedata=np.append(volumedata,distancemax)
    volumepile=np.append(volumepile,volumedata)
    print("\r{0}/{1}".format(i+1,samplerange),end="")
  print("")
  volumepile=np.reshape(volumepile,(samplerange,checking_freq_num))
  volumepile=volumepile**(0.5)
  velecitypile=np.array([],dtype=np.uint8)
  i=np.max(volumepile)
  if i!=0.0:
    volumepile=volumepile*127/i
    velocitypile=volumepile.astype(np.uint8)
  else:
    velocitypile=np.zeros((samplerange,checking_freq_num))
  outnote=np.array([],dtype=np.uint8)
  recorddelta=start_wait-61
  for i in range(samplerange):
    if np.max(velocitypile[i,:])==0:
      recorddelta+=div_delta
    else:
      subarray=np.array([],dtype=np.uint8)
      for j in range(checking_freq_num):
        if velocitypile[i,j]>0:
          subarray=np.append(subarray,DeltaTime(recorddelta))
          subarray=np.append(subarray,[144,noteset_array[j],velocitypile[i,j]])
          recorddelta=0
      recorddelta=div_delta
      for j in range(checking_freq_num):
        if velocitypile[i,j]>0:
          subarray=np.append(subarray,DeltaTime(recorddelta))
          subarray=np.append(subarray,[128,noteset_array[j],0])
          recorddelta=0
      outnote=np.append(outnote,subarray)
    print("\r{0}/{1}".format(i+1,samplerange),end="")
  print("")
  return outnote

def DeltaTime(delta):
  out=np.array([],dtype=np.uint8)
  out=np.append([delta%128],out)  #一番下のバイト
  delta=int(delta/128)
  if delta>0:
    out=np.append([128+delta%128],out)  #下から2番目バイト
    delta=int(delta/128)
  if delta>0:
    out=np.append([128+delta%128],out)  #下から3番目バイト
    delta=int(delta/128)
  if delta>0:
    out=np.append([128+delta],out)  #下から4番目バイト
  return out


def MakeMidi():
  #midi配列(uint8表記)
  midiout=np.array([],dtype=np.uint8)
  
  #ヘッダ
  midiout=np.append(midiout,[77,84,104,100])  #MThd
  midiout=np.append(midiout,[0,0,0,6])  #実データの大きさ(6)
  midiout=np.append(midiout,[0,1])  #フォーマット
  midiout=np.append(midiout,[0,2])  #トラック数
  midiout=np.append(midiout,[int(delta_time/256),delta_time%256])  #時間単位
  
  #トラック0(Conductor)
  midiout=np.append(midiout,[77,84,114,107])  #MThd
  track=np.array([],dtype=np.uint8)  #データ長を調べるために先をつくる
  i=int(60000000/tempo)
  track=np.append(track,[0,255,81,3,int(i/(256*256)),int(i/256)%256,i%256])  #テンポ
  track=np.append(track,[0,255,88,4,4,2,24,8])  #4/4拍子
  track=np.append(track,[0,255,47,0])  #トラック終端
  i=track.shape[0]
  midiout=np.append(midiout,[int(i/(256*256*256)),int(i/(256*256))%256,int(i/256)%256,i%256])  #データ長
  midiout=np.append(midiout,track)
  
  #トラック1(SineWave)
  midiout=np.append(midiout,[77,84,114,107])  #MThd
  track=np.array([],dtype=np.uint8)  #データ長を調べるために先をつくる
  track=np.append(track,[0,255,33,1,0])  #ポート指定
  track=np.append(track,[0,240,5,126,127,9,1,247])  #GMシステムオン
  track=np.append(track,[60,240,10,65,16,66,18,64,0,127,0,65,247])  #60dt後GSReset
  track=np.append(track,[0,176,121,0])  #ResetAllControl
  track=np.append(track,[1,176,0,8,0,176,32,0,0,192,80])  #1dt後PC:SineWave
  track=np.append(track,MakeNotes())  #ノーツを書き込む
  track=np.append(track,[0,255,47,0])  #トラック終端
  i=track.shape[0]
  midiout=np.append(midiout,[int(i/(256*256*256)),int(i/(256*256))%256,int(i/256)%256,i%256])  #データ長
  midiout=np.append(midiout,track)
  
  #midioutをバイナリにして出力
  outfile=b""
  outsec=b""
  j=midiout.shape[0]
  for i in range(j):
    outsec+=int(midiout[i]).to_bytes(1,byteorder="big")
    if i%binary_sec==binary_sec-1:
      outfile+=outsec
      outsec=b""
    print("\r{0}/{1}".format(i+1,j),end="")
  outfile+=outsec
  print("")
  args=sys.argv
  f=open(args[2],"wb")
  f.write(outfile)
  f.close()


if __name__ == '__main__':
  args=sys.argv
  check=np.array(args)
  if check.shape[0]<3:
    print("Error:コマンドライン引数が不適切です")
    sys.exit()
  if os.path.exists(args[1])==False:
    print("Error:指定したwavファイルが存在しません")
    sys.exit()
  MakeMidi()
