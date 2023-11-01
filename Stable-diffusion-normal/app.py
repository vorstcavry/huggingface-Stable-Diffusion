"""
Stable Diffusion Webui Version 1.6
https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.6.0

"""
commit_id=r"5ef669de080814067961f28357256e8fe27544f4" 
import os
from sys import executable 
import subprocess
import pathlib
import gc
import time
import subprocess

def Gitclone(URI:str,ClonePath:pathlib.Path ) -> int :
  if pathlib.Path.exists(ClonePath):
    return 0
  for z in range(10):
    i=subprocess.run([r"git",r"clone",str(URI),str(ClonePath)])
    if(i.returncode == 0 ): 
     del i
     return 0
    else :
     del i
  raise Exception(str.format("clone \'{0}\' failed",URI))
    

def DownLoad(URI:str,DownloadPath:pathlib.Path,DownLoadFileName:str ) -> int:
  if (DownloadPath / DownLoadFileName).is_file(): return 0
  for z in range(10):
    i=subprocess.run([r"aria2c",r"-c",r"-x" ,r"16", r"-s",r"16", r"-k" ,r"1M" ,r"-m",r"0",r"--enable-mmap=false",r"--console-log-level=error",r"-d",str(DownloadPath),r"-o",DownLoadFileName,URI]);
    if(i.returncode == 0 ): 
      del i
      gc.collect()
      return 0
    else :
      del i
  raise Exception(str.format("download \'{0}\' failed",URI))

user_home =pathlib.Path.home().resolve()
os.chdir(str(user_home))
#clone stable-diffusion-webui repo
print("cloning stable-diffusion-webui repo")
Gitclone(r"https://github.com/AUTOMATIC1111/stable-diffusion-webui.git",user_home / r"stable-diffusion-webui")
os.chdir(str(user_home / r"stable-diffusion-webui"))
os.system("git reset --hard "+commit_id)
os.chdir(user_home / r"stable-diffusion-webui")
Gitclone(r"https://github.com/vorstcavry/ncpt_colab_timer",user_home / r"stable-diffusion-webui" / r"extensions" / r"ncpt_colab_timer")
Gitclone(r"https://github.com/vorstcavry/static",user_home / r"stable-diffusion-webui" / r"static")

def run_echo_command():
    try:
        start_huggingface
    except NameError:
        start_huggingface = int(time.time()) - 5

    cmd = f"echo -n {start_huggingface} > /home/user/app/stable-diffusion-webui/static/colabTimer.txt"
    subprocess.run(cmd, shell=True)

# Contoh pemanggilan fungsi run_echo_command:
run_echo_command()
os.chdir(user_home / r"stable-diffusion-webui")
#install extensions
print("installing extensions")
#Gitclone(r"https://github.com/vorstcavry/embeddings",user_home / r"stable-diffusion-webui" / r"embeddings"  / r"negative")
#Gitclone(r"https://github.com/vorstcavry/lora",user_home / r"stable-diffusion-webui" / r"models" / r"Lora" / r"positive")
#Gitclone(r"https://github.com/vorstcavry/Checkpoint-Model",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint")

DownLoad(r"https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth",user_home / r"stable-diffusion-webui" / r"models" / r"ESRGAN" ,r"4x-UltraSharp.pth")
while (True):
   i=subprocess.run([r"wget",r"https://raw.githubusercontent.com/vorstcavry/s-scripts/main/run_n_times.py",r"-O",str(user_home / r"stable-diffusion-webui" / r"scripts" / r"run_n_times.py")])
   if(i.returncode == 0 ): 
    del i
    gc.collect()
    break
   else :
    del i
#Gitclone(r"https://github.com/deforum-art/deforum-for-automatic1111-webui",user_home / r"stable-diffusion-webui" / r"extensions" / r"deforum-for-automatic1111-webui" )
#Gitclone(r"https://github.com/AlUlkesh/stable-diffusion-webui-images-browser",user_home / r"stable-diffusion-webui" / r"extensions"/ r"stable-diffusion-webui-images-browser")
#Gitclone(r"https://github.com/camenduru/stable-diffusion-webui-huggingface",user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-huggingface")
Gitclone(r"https://github.com/BlafKing/sd-civitai-browser-plus",user_home / r"stable-diffusion-webui" / r"extensions" / r"civitai-browser")
#Gitclone(r"https://github.com/kohya-ss/sd-webui-additional-networks",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks")
Gitclone(r"https://github.com/Mikubill/sd-webui-controlnet",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-controlnet")
#Gitclone(r"https://github.com/fkunn1326/openpose-editor",user_home / r"stable-diffusion-webui" / r"extensions" / r"openpose-editor")
#Gitclone(r"https://github.com/jexom/sd-webui-depth-lib",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-depth-lib")
#Gitclone(r"https://github.com/hnmr293/posex",user_home / r"stable-diffusion-webui" / r"extensions" / r"posex")
#Gitclone(r"https://github.com/nonnonstop/sd-webui-3d-open-pose-editor",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-3d-open-pose-editor")
#Gitclone(r"https://github.com/hnmr293/posex",user_home / r"stable-diffusion-webui" / r"extensions" / r"posex")
Gitclone(r"https://github.com/vorstcavry/sd-webui-cloud-inference",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-cloud-inference")
Gitclone(r"https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git" , user_home / r"stable-diffusion-webui" / r"extensions" / r"a1111-sd-webui-tagcomplete")
#Gitclone(r"https://github.com/camenduru/sd-webui-tunnels",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-tunnels")
Gitclone(r"https://github.com/etherealxx/batchlinks-webui",user_home / r"stable-diffusion-webui" / r"extensions" / r"batchlinks-webui")
Gitclone(r"https://github.com/zanllp/sd-webui-infinite-image-browsing",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-infinite-image-browsing")
#Gitclone(r"https://github.com/catppuccin/stable-diffusion-webui",user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-catppuccin")
#Gitclone(r"https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg",user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-rembg")
Gitclone(r"https://tinyurl.com/aspect-ratio-v",user_home / r"stable-diffusion-webui" / r"extensions" / r"aspect-ratio")
#Gitclone(r"https://tinyurl.com/LOBE-Repo",user_home / r"stable-diffusion-webui" / r"extensions" / r"LOBE")
#Gitclone(r"https://github.com/hnmr293/sd-webui-llul",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-llul")
#Gitclone(r"https://github.com/IDEA-Research/DWPose",user_home / r"stable-diffusion-webui" / r"extensions" / r"DWPose")
#Gitclone(r"https://github.com/Bing-su/adetailer",user_home / r"stable-diffusion-webui" / r"extensions" / r"adetailer")
Gitclone(r"https://github.com/NoCrypt/sd_hf_out",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd_hf_out")

#Gitclone(r"https://github.com/Iyashinouta/sd-model-downloader",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-model-downloader")
#Gitclone(r"https://github.com/AIrjen/OneButtonPrompt",user_home / r"stable-diffusion-webui" / r"extensions" / r"OneButtonPrompt")
#Gitclone(r"https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards",user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-wildcards")
#Gitclone(r"https://github.com/adieyal/sd-dynamic-prompts",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-dynamic-prompts")
#Gitclone(r"https://github.com/d8ahazard/sd_dreambooth_extension",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd_dreambooth_extension")
#Gitclone(r"https://github.com/yfszzx/stable-diffusion-webui-inspiration",user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-inspiration")
#Gitclone(r"https://github.com/Coyote-A/ultimate-upscale-for-automatic1111",user_home / r"stable-diffusion-webui" / r"extensions" / r"ultimate-upscale-for-automatic1111")
os.chdir(user_home / r"stable-diffusion-webui")
#download ControlNet models
print("extensions dolwnload done .\ndownloading ControlNet models")
dList =[ r"https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors",
               r"https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
               r"https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
               r"https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
               r"https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors"]
for i in range(0,len(dList)): DownLoad(dList[i],user_home / r"stable-diffusion-webui" / r"models" / r"ControlNet",pathlib.Path(dList[i]).name)
del dList
#download ControlNet models
#print("extensions dolwnload done .\ndownloading ControlNet models")
#dList =[r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
##              r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors",#
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetenso#rs",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.sa#fetensors",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors#",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml"#,
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yam##l",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml",#
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_seg_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd14v1.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth",
#               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth"]
#for i in range(0,len(dList)): DownLoad(dList[i],user_home / r"stable-diffusion-webui" / r"extensions" / "sd-webui-controlnet" / r"models",pathlib.Path(dList[i]).name)
#del dList
#d#ownload model    
#you can change model download address here
#print("ControlNet models download done.\ndownloading model")
#Stable Diffusion Checkpoint Model
#anything version4.5
#DownLoad(r"https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.5-pruned.ckpt",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"anything-v4.5-pruned.ckpt")
#DownLoad(r"https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.0.vae.pt",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"anything-v4.0.vae.pt")
#Counterfeit-V3.0
#DownLoad(r"https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fp16.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"Counterfeit-V3.0_fp16.safetensors")
#AbyssOrangeMix2 sfw
#DownLoad(r"https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_sfw.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"AbyssOrangeMix2_sfw.safetensors")
#DownLoad(r"https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"orangemix.vae.pt")
#MeinaPastelV5
#DownLoad(r"https://huggingface.co/Meina/MeinaPastel/resolve/main/MeinaPastelV5%20-%20Baked%20VAE.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"MeinaPastelV5_BakedVAE.safetensors")
#DownLoad(r"https://huggingface.co/AnonPerson/ChilloutMix/resolve/main/ChilloutMix-ni-fp16.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"ChilloutMix-ni-fp16.safetensors")
#DownLoad(r"https://huggingface.co/Meina/MeinaPastel/resolve/main/MeinaPastelV4%20-%20Without%20VAE.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"MeinaPastelV4%20-%20Without%20VAE.safetensors")
#DownLoad(r"https://huggingface.co/ckpt/perfect_world/resolve/main/perfectWorld_v2Baked.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"perfectWorld_v2Baked.safetensors")
#DownLoad(r"https://huggingface.co/vorstcavry/figurestyle1/resolve/main/figure.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"figure.safetensors")
#DownLoad(r"https://huggingface.co/vorstcavry/dosmix/resolve/main/ddosmix_V2.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"ddosmix_V2.safetensors")
#DownLoad(r"https://huggingface.co/ckpt/rev-animated/resolve/main/revAnimated_v11.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"revAnimated_v11.safetensors")
#DownLoad(r"https://huggingface.co/ckpt/MeinaMix/resolve/main/Meina_V8_baked_VAE.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"Meina_V8_baked_VAE.safetensors")
#DownLoad(r"https://huggingface.co/ckpt/CyberRealistic/resolve/main/cyberrealistic_v13.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"cyberrealistic_v13.safetensors")
DownLoad(r"https://huggingface.co/vorstcavry/mymodel/resolve/main/Cavry_V2.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion" / r"Checkpoint",r"Cavry_V2.safetensors")
#downloadvae
DownLoad(r"https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",user_home / r"stable-diffusion-webui" / r"models" / r"VAE",r"vae-ft-mse-840000-ema-pruned.safetensors")

#Lora Model
#Better Light
#DownLoad(r"https://civitai.com/api/download/models/39885",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora",r"Better_light.safetensors")
#DownLoad(r"https://civitai.com/api/download/models/39885",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"Better_light.safetensors")
#LAS
#DownLoad(r"https://civitai.com/api/download/models/21065",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora",r"LAS.safetensors")
#DownLoad(r"https://civitai.com/api/download/models/21065",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"LAS.safetensors")
#Backlighting
#DownLoad(r"https://civitai.com/api/download/models/39164",user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora",r"backlighting.safetensors")
#DownLoad(r"https://civitai.com/api/download/models/39164",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"backlighting.safetensors")
#DownLoad(r"https://huggingface.co/vorstcavry/loraasia1/resolve/main/japaneseDollLikeness_v15.safetensors",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"japaneseDollLikeness_v15.safetensors")
#DownLoad(r"https://huggingface.co/vorstcavry/loraasia1/resolve/main/koreanDollLikeness_v20.safetensors",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"koreanDollLikeness_v20.safetensors")
#DownLoad(r"https://huggingface.co/vorstcavry/loraasia1/resolve/main/taiwanDollLikeness_v15.safetensors",user_home / r"stable-diffusion-webui" /  r"models"/ r"lora",r"taiwanDollLikeness_v15.safetensors")


DownLoad(r"https://huggingface.co/vorstcavry/test/resolve/main/SD-A/ui-config.json",user_home / r"stable-diffusion-webui",r"ui-config.json")
DownLoad(r"https://huggingface.co/vorstcavry/test/resolve/main/SD-A/config.json",user_home / r"stable-diffusion-webui",r"ui-config.json")


#GFPGAN Model
#detection Resnet50
DownLoad(r"https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",user_home / r"stable-diffusion-webui"/r"models"/r"GFPGAN",r"detection_Resnet50_Final.pth")
#parsing_parsenet
DownLoad(r"https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",user_home / r"stable-diffusion-webui"/r"models"/r"GFPGAN",r"parsing_parsenet.pth")
#GFPGANv1.4
DownLoad(r"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",user_home / r"stable-diffusion-webui"/r"models"/r"GFPGAN",r"GFPGANv1.4.pth")
#strt Stable Diffusion Webui
print("Done\nStarting Webui...")
os.chdir(user_home / r"stable-diffusion-webui")
import subprocess
import pathlib

user_home = pathlib.Path("/home/user")  # Gantilah dengan path yang sesuai

args = [
    executable,
    user_home / "stable-diffusion-webui" / "launch.py",
    "--precision", "full",
    "--no-half",
    "--no-half-vae",
    "--enable-insecure-extension-access",
    "--medvram",
    "--skip-torch-cuda-test",
    "--enable-console-prompts",
    "--ui-settings-file=" + str(pathlib.Path(__file__).parent / "config.json"),
    "--hf-token-out",
    "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"  # Gantilah dengan token yang sesuai
]

args = [arg.as_posix() if isinstance(arg, pathlib.PosixPath) else arg for arg in args]

try:
    ret = subprocess.run(args)
except Exception as e:
    print("Error:", e)
del os ,user_home ,pyexecutable ,subprocess