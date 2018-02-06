# -*- coding: utf-8 -*-
import os
import glob
import shutil

#パスを確認する、なければ作る
def MakeFolder(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        return True
    return False

#フォルダ/ファイルを削除
def RmFolder(fpath):
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
        return True
    return False

def RmAndMakeFolder(fpath):
    if os.path.exists(fpath):
        RmFolder(fpath)
    MakeFolder(fpath)

#フォルダ内のファイルを取得
def getFileName(fpath, ext):
    if not os.path.exists(fpath):
        files = glob.glob(os.path.join(fpath, '*.'+ext))
        return [os.path.split(v)[1] for v in files]
    return False