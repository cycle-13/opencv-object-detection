# Description: Quick youtube video downloader. Nothing special.
# Written by: Jp C Aldama
# Date: 3/7/2020 - 4:46 PM
# Usage: without qoutes or angle brackets 'python tubejackal.py <URL TO YOUTUBE VIDEO>'
# License: MIT 
# Written by: JP Aldama (Quaxis Corporation 2020)
# Requirements: pip install pytube3
# TODO Time processes, click cli, queued downloads, playlist downloads 
# 	Mp3 ripping, download a chunk of a video

# Usage [enter exact, no quotes or anglebrackets]: 
# python tubejackal.py https://youtubeurl
# Special Thanks to pytube3 author https://twitter.com/hmartin

from pytube import YouTube
import os, sys

url = sys.argv[1]

def jack_video(url):
    yt = YouTube(url)
    yt.streams[0].download()
    title = yt.title
    print(f"[-TUBEJACKAL-] {title} has sucessfully downloaded!")

if __name__ == '__main__':
    jack_video(url)
