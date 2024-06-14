import sys 
sys.path.append("./..")
import json 
import csv 
import pandas as pd 
import numpy as np 
import torch 
from tqdm import tqdm 
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)

def split_json(filename):
    with open(filename, "r") as f:
        contents = json.load(f)
        for i, expl in tqdm(enumerate(contents)):
            dumped = json.dumps(expl)
            with open(f"./../../expl/background/gradcam/gradcam_cnn_train_{i}.json", "w") as f:
                f.write(dumped)

def background_seg():
    from dataset.loader import getLoader
    from PIL import Image
    import torchvision 
    datasets, _ = getLoader("background", "/nvmescratch/seongmin/hfb/data/background", 1, 1)
    datasets[0].seg_transform = None
    for data in datasets[0]:
        print(data["img_name"])
        seg_filename = data["img_name"].replace("original", "only_fg")
        print(type(data["seg"]), data["seg"].size)
        print(data["seg"])
        exit()

def manually_label():
    fb_filename = "./../../feedback/waterbirds/gradcam_resnet50_train_1.json"
    num_train = 4795
    fb = ["maybe"] * num_train
    manual_label = [
        ""
    ]
    # pass

def generate_attention_mturk_csv():
    import csv 

    indices = range(4000, 4795)
    # indices = [6,7,9,10,11,12,81,85,86,90]
    template = "https://24-crayon-mturk-waterbirds.s3.amazonaws.com/crayon-attention/gradcam_{}/gradcam_bicubic_{}.png"

    with open("mturk-attention-waterbirds-feb23-10am.csv", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["image_url"])
        for i in indices:
            csvwriter.writerow([template.format(i//1000, i)])

def generate_attention_mturk_result_json():
    mturk_result_dir = "./../../feedback/waterbirds/mturk/attention"
    fb_filename = os.path.join(mturk_result_dir, "mturk_attention_feb23_10am.json")
    df = pd.DataFrame()
    for filename in os.listdir(mturk_result_dir):
        if not filename.endswith("csv"): continue
        mturk_result_filename = os.path.join(mturk_result_dir, filename)
        mturk_result_df = pd.read_csv(mturk_result_filename)[["Input.image_url", "Answer.category.label"]]
        df = pd.concat([df, mturk_result_df])
    df["Input.image_url"] = df["Input.image_url"].apply(lambda x: int(x.split("_")[-1].split(".")[0]))
    df["Answer.category.label"] = df["Answer.category.label"].apply(lambda x: 1 if x=="Yes" else 0)
    df = df.rename(columns={"Input.image_url": "image_idx", "Answer.category.label": "hfb"})
    df = df.sort_values("image_idx")
    df = df.reset_index(drop=True)
    df = df.groupby(["image_idx"]).mean()
    df["hfb"] = df["hfb"].apply(lambda x: "yes" if x>0.7 else ("maybe" if x>=0.5 else "no"))
    print(len(df))

    num_train = 4795
    fb = np.array(["maybe"] * num_train)
    for image_idx, row in df.iterrows():
        hfb = row["hfb"]
        fb[image_idx] = hfb 
    
    dumped = json.dumps(fb, cls=NumpyEncoder)
    with open(fb_filename, "w") as f: f.write(dumped)

def generate_pruning_mturk_csv():
    import csv 
    import json 
    
    with open("./../../expl/waterbirds/neuron_concept_cluster_1.json", "r") as f:
        neuron_cluster = json.load(f)
    
    neuron_numbers = [i for i in range(0,2048)]

    cls_img_coord_names = []
    for neuron in neuron_numbers:
        for (k,v) in neuron_cluster.items():
            if neuron in v:
                patches = k.split(", (")
                patches = [[p.strip("[()] '") for p in patch.split(",")] for patch in patches]
                coords = [(int(float(patch[0])), int(float(patch[1]))) for patch in patches] 
                cls_nums = [patch[2].split(".")[0] for patch in patches]
                img_nums = ["_".join(patch[2].rstrip(".jpg").split("_")[-2:]) for patch in patches]
                for coord, cls_num, img_num in zip(coords, cls_nums, img_nums):
                    cls_img_coord_names.append((cls_num, img_num, coord))
    cls_img_coord_names = cls_img_coord_names

    # generate already_tested
    mturk_result_dir = "./../../feedback/waterbirds/mturk/pruning"
    df = pd.DataFrame()
    for filename in os.listdir(mturk_result_dir):
        if not filename.endswith("csv"): continue
        mturk_result_filename = os.path.join(mturk_result_dir, filename)
        mturk_result_df = pd.read_csv(mturk_result_filename)[["Input.image_url"]]
        df = pd.concat([df, mturk_result_df])
    already_tested = df["Input.image_url"].tolist()

    vis_dir = "./../../vis/waterbirds/mturk_concept_1"
    template = "https://24-crayon-mturk-waterbirds.s3.amazonaws.com/crayon-pruning/"
    filename_template = "patch_cls_{}_img_{}_coord_{}.png"

    filenames = []
    for cls_img_coord in cls_img_coord_names:
        filename = filename_template.format(cls_img_coord[0], cls_img_coord[1], f"{cls_img_coord[2][0]}_{cls_img_coord[2][1]}")
        filenames.append(filename)
    filenames = list(set(filenames))

    # print(already_tested)
    num_patches = 0
    with open("mturk-pruning-waterbirds-feb24-10am.csv", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["image_url"])
        for filename in filenames:
            if not os.path.exists(os.path.join(vis_dir, filename)):
                raise ValueError(f"File {os.path.join(vis_dir, filename)} doesn't exist")
            if template+filename not in already_tested: 
                csvwriter.writerow([template+filename])
                num_patches += 1
            else:
                # print(template+filename, "already tested")
                pass

    print(num_patches, "patches added")

def generate_pruning_mturk_result_json():
    expl_dir = "./../../expl/waterbirds"
    fb_dir = "./../../feedback/waterbirds"
    mturk_result_dir = os.path.join(fb_dir, "mturk/pruning")
    fb_filename = os.path.join(mturk_result_dir, "final.json")
    df = pd.DataFrame()
    for filename in os.listdir(mturk_result_dir):
        if not filename.endswith("csv"): continue
        mturk_result_filename = os.path.join(mturk_result_dir, filename)
        mturk_result_df = pd.read_csv(mturk_result_filename)[["Input.image_url", "Answer.category.label"]]
        df = pd.concat([df, mturk_result_df])
    df["img_name"] = df["Input.image_url"].apply(lambda x: x.split("/")[-1])
    df = df.rename(columns={"Answer.category.label": "hfb"})
    print(len(df))

    img_to_neuron_filename = "img_to_neuron_resnet50_train_1.json"
    neuron_to_img_filename = "neuron_to_img_resnet50_train_1.json"
    with open(os.path.join(expl_dir, img_to_neuron_filename), "r") as f: img_to_neuron_dict = json.load(f)
    with open(os.path.join(expl_dir, neuron_to_img_filename), "r") as f: neuron_to_img_dict = json.load(f)

    num_neuron = 2048
    neuron_vote = np.zeros([num_neuron])
    neuron_visited = np.zeros([num_neuron])
    img_responses = dict()
    for img_idx, row in df.iterrows():
        hfb_val = 1 if row["hfb"].lower()=="yes" else -1
        img_name = row["img_name"]
        img_responses[img_name] = (row["hfb"].lower()=="yes")
        assert len(img_to_neuron_dict[img_name]) > 0
        # print(img_name, len(img_to_neuron_dict[img_name]))
        for neuron in img_to_neuron_dict[img_name]:
            neuron_vote[neuron] += hfb_val 
            neuron_visited[neuron] += 1
    mturk_neurons = np.where(neuron_visited!=0)[0]
    print("Visited:", len(mturk_neurons))
    mturk_neurons = np.where(neuron_visited==3)[0]
    print("Fully visited:", len(mturk_neurons))
    # print(mturk_neurons)

    fbs = (neuron_vote >= 0)  # majority vote
    # fbs = (neuron_vote == 3)  # should get 3 yes to be annotated as "yes"

    with open(os.path.join(fb_dir, "concept_resnet50_train_1.json"), "r") as f: gt_concept_label = json.load(f)
    with open(os.path.join(expl_dir, "neuron_concept_cluster_1.json"), "r") as f: neuron_concept_cluster = json.load(f)

    # TODO: delete this later
    # for neuron in range(2048): if not gt_concept_label[neuron]: fbs[neuron]=False
    
    dumped = json.dumps(fbs, cls=NumpyEncoder)
    with open(fb_filename, "w") as f: f.write(dumped)

    # manually fix gt label
    # gt_concept_label = np.array(gt_concept_label)
    # gt_concept_label[6] = True 
    # gt_concept_label[33] = True 
    # gt_concept_label[40] = True 
    # gt_concept_label[47] = True 
    # gt_concept_label[51] = True 
    # gt_concept_label[52] = True 
    # gt_concept_label[55] = True 
    # gt_concept_label[58] = True 
    # gt_concept_label[60] = True 
    # gt_concept_label[67] = True 
    # gt_concept_label[76] = True 
    # gt_concept_label[83] = True 
    # gt_concept_label[93] = True 
    # gt_concept_label[199] = False 
    # gt_concept_label[273] = False 
    # gt_concept_label[348] = True
    # gt_concept_label[877] = False
    # gt_concept_label[944] = True
    # gt_concept_label[1206] = True
    # gt_concept_label[1370] = True
    # gt_concept_label[1456] = False
    # gt_concept_label[1582] = True
    # gt_concept_label[1610] = True
    # gt_concept_label[[148, 354, 445, 463, 464, 475, 698, 720, 771, 1084, 1184, 1190, 1282, 1534, 1541, 1652, 1674, 1748, 1813, 1815, 1990]] = True
    # 66, 1453, 81, 91 is wrongly labeled but understandable
    # [170, 315, 1930]  is wrongly labeled but understandable
    # gt_concept_label[[66, 1453, 81, 91, 170, 315, 1930]] = False


    gt_label = np.array(gt_concept_label)[mturk_neurons]
    mturk_label = np.array(fbs)[mturk_neurons]
    num_diff = 0
    wrong_imgs = []
    allowed_wrong_imgs = [
        ("patch_cls_101_img_0056_97071_coord_6_3.png", True),
        ("patch_cls_184_img_0077_795247_coord_3_2.png", True),
        ("patch_cls_088_img_0040_77823_coord_4_3.png", True),
        ("patch_cls_025_img_0037_23889_coord_3_3.png", True),
        ("patch_cls_073_img_0099_64435_coord_2_5.png", True),
        ("patch_cls_095_img_0007_88038_coord_2_3.png", True),
        ("patch_cls_151_img_0043_797458_coord_2_5.png", True),
        ("patch_cls_127_img_0118_118603_coord_2_2.png", True),
        ("patch_cls_122_img_0028_116656_coord_5_6.png", False),
        ("patch_cls_151_img_0043_797458_coord_1_5.png", True),
        ("patch_cls_192_img_0102_184263_coord_3_3.png", True),
        ("patch_cls_051_img_0072_34497_coord_3_3.png", True),
        ("patch_cls_085_img_0079_73958_coord_2_4.png", True),
        ("patch_cls_005_img_0063_794901_coord_1_1.png", True),
        ("patch_cls_129_img_0010_121331_coord_1_4.png", True),
        ("patch_cls_187_img_0008_179850_coord_3_5.png", False),
        ("patch_cls_127_img_0014_120072_coord_2_5.png", True),
        ("patch_cls_097_img_0098_91401_coord_3_2.png", False),
        ("patch_cls_129_img_0010_121331_coord_1_5.png", False), # ambigous; only small beak part is in the patch
        ("patch_cls_117_img_0064_110664_coord_2_5.png", True), # ambigous; only small beak part is in the patch
        ("patch_cls_171_img_0009_166752_coord_2_3.png", True),
        ("patch_cls_126_img_0055_796937_coord_4_4.png", True),
        ("patch_cls_095_img_0007_88038_coord_1_3.png", True),
        ("patch_cls_129_img_0010_121331_coord_2_5.png", False),  # can be ambiguous; for me it looks True but labeled as False
        ("patch_cls_133_img_0097_128967_coord_2_5.png", False),  # can be ambiguous
        ("patch_cls_095_img_0007_88038_coord_0_2.png", False),  # can be ambiguous
        ("patch_cls_169_img_0102_165884_coord_1_0.png", True),  # Very small tail part looks to be in the patch but almost obvious false
        ("patch_cls_125_img_0046_117405_coord_3_4.png", True),
        ("patch_cls_175_img_0025_171110_coord_3_2.png", True),
        ("patch_cls_179_img_0067_174999_coord_3_5.png", True),  # ambiguous; only a small tail part
        ("patch_cls_040_img_0076_30555_coord_3_6.png", False),
        ("patch_cls_184_img_0077_795247_coord_2_2.png", True),
        ("patch_cls_091_img_0096_79878_coord_4_3.png", True),  # ambiguous; very small leg part is in the patch
        ("patch_cls_137_img_0094_133114_coord_3_2.png", True),
        ("patch_cls_119_img_0011_113420_coord_4_3.png", False),  # ambiguos
        ("patch_cls_074_img_0079_64713_coord_6_6.png", False),
        ("patch_cls_157_img_0066_795007_coord_3_3.png", True),
        ("patch_cls_104_img_0031_99607_coord_2_2.png", True),
        ("patch_cls_088_img_0069_78587_coord_0_0.png", False),
        ("patch_cls_073_img_0078_62351_coord_2_4.png", False),  # can be ambiguous; I would have marked as true
        ("patch_cls_095_img_0007_88038_coord_2_2.png", False),  # can be ambiguous
        ("patch_cls_041_img_0110_42136_coord_2_2.png", True),
        ("patch_cls_128_img_0025_796518_coord_1_6.png", False),
        ("patch_cls_174_img_0024_170501_coord_2_4.png", True),
        ("patch_cls_019_img_0006_20867_coord_4_2.png", True),
        ("patch_cls_070_img_0025_795692_coord_4_4.png", True),
        ("patch_cls_182_img_0119_176485_coord_2_3.png", True),
        ("patch_cls_115_img_0068_107422_coord_5_3.png", True),
        ("patch_cls_007_img_0028_795944_coord_3_1.png", False),
        ("patch_cls_171_img_0050_166820_coord_3_4.png", True),
        ("patch_cls_085_img_0079_73958_coord_3_2.png", True),
        ("patch_cls_128_img_0025_796518_coord_1_5.png", False),
        ("patch_cls_126_img_0067_796912_coord_5_2.png", True),
        ("patch_cls_128_img_0025_796518_coord_1_5.png", False),
        ("patch_cls_018_img_0008_796789_coord_5_1.png", True),  # ambiguous, only a tail part
        ("patch_cls_097_img_0009_91902_coord_3_2.png", True),
        ("patch_cls_117_img_0066_110819_coord_1_5.png", False),
        ("patch_cls_053_img_0093_36193_coord_3_5.png", False),  # don't agree but make sense
        ("patch_cls_017_img_0056_18352_coord_5_5.png", True),  # don't agree but make sense
        ("patch_cls_024_img_0036_796329_coord_1_0.png", True),  # don't agree but make sense
        ("patch_cls_009_img_0070_2325_coord_6_6.png", True),  # don't agree but make sense
        ("patch_cls_063_img_0052_50013_coord_6_6.png", True),  # don't agree but make sense
        ("patch_cls_195_img_0122_186365_coord_1_0.png", True),  # don't agree but make sense
        ("patch_cls_032_img_0029_794624_coord_5_1.png", True),
        ("patch_cls_003_img_0045_1162_coord_1_0.png", True),
        ("patch_cls_110_img_0112_104548_coord_5_5.png", True),  # ambiguous
        ("patch_cls_066_img_0062_53538_coord_1_0.png", True),
        ("patch_cls_052_img_0113_35703_coord_0_1.png", True),
        ("patch_cls_031_img_0093_795316_coord_5_5.png", True),  # don't agree but make sense
        ("patch_cls_020_img_0032_21823_coord_1_0.png", True),  # ambiguous
        ("patch_cls_127_img_0118_118603_coord_1_1.png", True),
        ("patch_cls_112_img_0017_797028_coord_0_0.png", True),  # ambiguous
        ("patch_cls_103_img_0113_98630_coord_5_5.png", True),  # ambiguous
        ("patch_cls_127_img_0014_120072_coord_1_5.png", True),
        ("patch_cls_153_img_0029_794760_coord_0_6.png", True),
        ("patch_cls_184_img_0023_795269_coord_1_6.png", True),
        ("patch_cls_061_img_0098_45753_coord_2_3.png", True),
        ("patch_cls_117_img_0106_797247_coord_1_2.png", True),
        ("patch_cls_114_img_0086_106970_coord_4_4.png", True),
        ("patch_cls_073_img_0003_63408_coord_4_3.png", True),
        ("patch_cls_014_img_0055_13473_coord_4_3.png", True),
        ("patch_cls_170_img_0072_795379_coord_2_4.png", True),
        ("patch_cls_119_img_0029_113434_coord_4_2.png", True),
        ("patch_cls_144_img_0030_147825_coord_4_4.png", True),  # ambiguous
        ("patch_cls_025_img_0037_23889_coord_4_3.png", True),
        ("patch_cls_015_img_0089_14598_coord_3_2.png", True),
        ("patch_cls_004_img_0065_1502_coord_4_4.png", True),  # don't agree but make sense
        ("patch_cls_109_img_0077_104042_coord_5_3.png", True),  # don't agree but make sense
        ("patch_cls_075_img_0066_65902_coord_2_5.png", True),
        ("patch_cls_017_img_0051_17223_coord_4_2.png", True),
        ("patch_cls_018_img_0012_796802_coord_5_3.png", True),
        ("patch_cls_159_img_0095_160406_coord_1_4.png", True),  # ambiguous (beak)
        ("patch_cls_118_img_0092_111413_coord_2_4.png", True),
        ("patch_cls_144_img_0076_148391_coord_1_3.png", True),
        ("patch_cls_135_img_0062_129548_coord_3_3.png", True),
        ("patch_cls_030_img_0022_26062_coord_5_4.png", True),  # ambiguous (leg)
        ("patch_cls_143_img_0049_145755_coord_2_3.png", True),
        ("patch_cls_187_img_0001_179912_coord_4_3.png", True),
        ("patch_cls_081_img_0080_72199_coord_3_2.png", True),  # ambiguous (beak)
        ("patch_cls_192_img_0044_184170_coord_4_3.png", True),
        ("patch_cls_174_img_0003_170474_coord_3_4.png", True),  # ambiguous
        ("patch_cls_002_img_0088_883_coord_1_2.png", True),
        ("patch_cls_088_img_0098_78382_coord_5_4.png", True),
        ("patch_cls_021_img_0079_22690_coord_3_5.png", True),
        ("patch_cls_088_img_0095_78568_coord_4_4.png", True),
        ("patch_cls_001_img_0038_212_coord_2_2.png", True),  # ambiguous
        ("patch_cls_120_img_0093_114757_coord_4_2.png", True),
        ("patch_cls_114_img_0010_107375_coord_5_2.png", True),  # ambiguous; tail
        ("patch_cls_015_img_0009_15163_coord_3_3.png", True),
        ("patch_cls_065_img_0081_53264_coord_4_3.png", True),
        ("patch_cls_035_img_0092_27264_coord_4_3.png", True),
        ("patch_cls_015_img_0008_15195_coord_4_2.png", True),
        ("patch_cls_143_img_0072_147667_coord_2_2.png", True),
        ("patch_cls_199_img_0026_189816_coord_2_3.png", True),
        ("patch_cls_047_img_0091_31938_coord_2_4.png", True),
        ("patch_cls_113_img_0035_106904_coord_1_4.png", True),  # ambiguous; beak
        ("patch_cls_079_img_0053_70899_coord_2_2.png", True),
        ("patch_cls_113_img_0010_794575_coord_2_4.png", True), # ambiguous; beak
        ("patch_cls_025_img_0024_23712_coord_3_4.png", True),
        ("patch_cls_099_img_0003_92910_coord_1_4.png", True), # ambiguous; beak
        ("patch_cls_120_img_0053_114780_coord_4_3.png", True), # ambiguous; leg
        ("patch_cls_030_img_0068_25859_coord_1_3.png", True),
        ("patch_cls_011_img_0101_6880_coord_3_4.png", True),
        ("patch_cls_058_img_0058_40184_coord_3_1.png", True),
        ("patch_cls_039_img_0047_30393_coord_1_2.png", True),
        ("patch_cls_008_img_0030_797509_coord_4_3.png", True),  # ambiguous
        ("patch_cls_048_img_0008_33153_coord_4_4.png", True),
        ("patch_cls_052_img_0063_35529_coord_4_4.png", True),
        ("patch_cls_148_img_0054_154938_coord_5_3.png", True), # ambiguous; leg
        ("patch_cls_088_img_0069_78587_coord_5_2.png", True), 
        ("patch_cls_118_img_0046_112845_coord_4_2.png", True),
        ("patch_cls_154_img_0106_157102_coord_2_3.png", True),
        ("patch_cls_003_img_0072_796371_coord_2_4.png", True),
        ("patch_cls_159_img_0065_160111_coord_3_3.png", True),
        ("patch_cls_190_img_0052_794752_coord_3_5.png", True),
        ("patch_cls_177_img_0070_174650_coord_3_4.png", True),  # ambiguous; beak
        ("patch_cls_178_img_0041_794893_coord_5_4.png", True),  # ambiguous; leg
        ("patch_cls_145_img_0066_150864_coord_3_2.png", True),
        ("patch_cls_093_img_0101_85656_coord_2_2.png", True),
        ("patch_cls_168_img_0031_165363_coord_3_3.png", True),
        ("patch_cls_157_img_0066_795007_coord_5_2.png", True),  # ambiguous; leg
        ("patch_cls_098_img_0005_92362_coord_3_3.png", True),
        ("patch_cls_018_img_0036_19406_coord_4_6.png", True),
        ("patch_cls_195_img_0095_186561_coord_5_3.png", True),  # ambiguous; leg
        ("patch_cls_092_img_0019_83850_coord_4_2.png", True),
        ("patch_cls_125_img_0109_117940_coord_3_1.png", True),
        ("patch_cls_076_img_0103_67700_coord_5_4.png", True),  # ambiguous; leg
        ("patch_cls_184_img_0020_795265_coord_3_5.png", True),  # ambiguous
        ("patch_cls_043_img_0020_795482_coord_3_2.png", True),
        ("patch_cls_123_img_0018_116834_coord_2_5.png", True),  # ambiguous; beak
        ("patch_cls_097_img_0032_91201_coord_4_4.png", True),  # ambiguous; beak
        ("patch_cls_159_img_0022_160512_coord_3_3.png", True),
        ("patch_cls_195_img_0035_186356_coord_3_4.png", True),
        ("patch_cls_183_img_0105_176970_coord_3_4.png", True),  # beak;
        ("patch_cls_058_img_0005_40375_coord_5_3.png", True),  # ambiguous; tail
        ("patch_cls_184_img_0029_795262_coord_4_4.png", True), 
        ("patch_cls_101_img_0013_96901_coord_1_3.png", True),
        ("patch_cls_064_img_0108_51108_coord_3_4.png", True),
        ("patch_cls_001_img_0089_796069_coord_3_3.png", True),
        ("patch_cls_145_img_0052_150695_coord_1_2.png", True),
        ("patch_cls_120_img_0119_115512_coord_2_3.png", True),  # ambiguous; head
        ("patch_cls_161_img_0044_161760_coord_3_4.png", True),  # ambiguous; beak
        ("patch_cls_093_img_0073_85343_coord_5_4.png", True),  # ambiguous; leg
        ("patch_cls_184_img_0077_795247_coord_5_3.png", True),  # ambiguous; leg
        ("patch_cls_007_img_0035_795934_coord_3_5.png", True),
        ("patch_cls_171_img_0112_166754_coord_2_4.png", True),  # ambiguous; beak
        ("patch_cls_164_img_0087_163451_coord_2_3.png", True),
        ("patch_cls_184_img_0044_177526_coord_4_3.png", True),
        ("patch_cls_163_img_0031_163012_coord_4_4.png", True),
        ("patch_cls_108_img_0036_797359_coord_4_3.png", True),
        ("patch_cls_172_img_0079_167213_coord_3_5.png", True),
        ("patch_cls_137_img_0022_133786_coord_4_3.png", True),
        ("patch_cls_175_img_0017_171678_coord_5_3.png", True),  # ambiguous;leg
        ("patch_cls_115_img_0041_796711_coord_3_3.png", True),
        ("patch_cls_100_img_0131_94464_coord_5_2.png", True),  # ambiguous
        ("patch_cls_172_img_0092_167457_coord_5_2.png", True),  # tail
        ("patch_cls_007_img_0028_795944_coord_2_0.png", False),
        ("patch_cls_183_img_0020_177149_coord_1_1.png", False),
        ("patch_cls_164_img_0084_797177_coord_4_3.png", False),  # ambiguous; contain leg
        ("patch_cls_053_img_0093_36193_coord_2_6.png", False),  # beak
        ("patch_cls_007_img_0028_795944_coord_2_1.png", False),
        ("patch_cls_007_img_0028_795944_coord_3_0.png", False),
        ("patch_cls_025_img_0037_23889_coord_4_4.png", False),  # leg
        ("patch_cls_053_img_0032_36439_coord_1_6.png", False),
        ("patch_cls_001_img_0019_796104_coord_4_4.png", False),  # tail
        ("patch_cls_122_img_0028_116656_coord_6_6.png", False),
        ("patch_cls_022_img_0008_796968_coord_4_1.png", False),
        ("patch_cls_037_img_0007_795600_coord_6_5.png", False),
        ("patch_cls_151_img_0043_797458_coord_3_5.png", False),  # ambiguous
        ("patch_cls_040_img_0076_30555_coord_6_6.png", False),
        ("patch_cls_129_img_0036_121679_coord_3_0.png", False),
        ("patch_cls_187_img_0037_796166_coord_2_2.png", True),
        ("patch_cls_038_img_0006_29362_coord_3_2.png", True),
        ("patch_cls_036_img_0110_28602_coord_1_2.png", True),
        ("patch_cls_029_img_0113_25149_coord_5_3.png", True),  # ambiguous
        ("patch_cls_127_img_0118_118603_coord_1_2.png", True),
        ("patch_cls_034_img_0010_27039_coord_2_1.png", True),
        ("patch_cls_142_img_0090_143880_coord_5_2.png", True),  # ambiguous
        ("patch_cls_075_img_0071_65799_coord_1_3.png", True),
        ("patch_cls_046_img_0076_31639_coord_5_4.png", True),
        ("patch_cls_047_img_0100_32183_coord_4_5.png", True),
        ("patch_cls_178_img_0038_794882_coord_0_3.png", True),
        ("patch_cls_145_img_0076_150519_coord_2_1.png", True),
        ("patch_cls_035_img_0011_27633_coord_5_3.png", True),  # ambiguous; leg
        ("patch_cls_002_img_0025_571_coord_5_4.png", True),
        ("patch_cls_150_img_0033_155511_coord_5_3.png", True),  # leg
        ("patch_cls_195_img_0122_186365_coord_3_5.png", True),
        ("patch_cls_089_img_0049_79136_coord_5_4.png", True),  # ambiguous
        ("patch_cls_139_img_0055_138186_coord_5_2.png", True),  # leg
        ("patch_cls_189_img_0086_181891_coord_5_4.png", True),
        ("patch_cls_127_img_0014_120072_coord_1_4.png", True),
        ("patch_cls_172_img_0110_167268_coord_1_4.png", True),
        ("patch_cls_070_img_0089_795698_coord_1_2.png", True),
        ("patch_cls_155_img_0030_158488_coord_4_1.png", True),
        ("patch_cls_104_img_0019_99810_coord_5_2.png", True), # leg
        ("patch_cls_044_img_0004_43221_coord_1_2.png", True),
        ("patch_cls_120_img_0009_114796_coord_4_1.png", True), # ambiguous; tail
        ("patch_cls_108_img_0003_797376_coord_2_1.png", True), # ambiguous
        ("patch_cls_191_img_0094_183401_coord_5_3.png", True),  # ambiguous
        ("patch_cls_065_img_0006_796014_coord_5_4.png", True), # ambiguous
        ("patch_cls_036_img_0110_28602_coord_1_0.png", True), # very ambiguous
        ("patch_cls_095_img_0007_88038_coord_1_2.png", True),
        ("patch_cls_094_img_0055_86075_coord_1_2.png", True), # tail
        ("patch_cls_108_img_0003_797376_coord_5_3.png", True), # leg
        ("patch_cls_146_img_0066_151478_coord_5_3.png", True), # leg
        ("patch_cls_194_img_0030_185798_coord_4_1.png", True),
        ("patch_cls_188_img_0110_180521_coord_4_6.png", True),
        ("patch_cls_048_img_0053_794639_coord_5_2.png", True),
        ("patch_cls_060_img_0132_44435_coord_5_4.png", True), #leg
        ("patch_cls_025_img_0012_23565_coord_5_3.png", True),
        ("patch_cls_129_img_0092_121969_coord_2_4.png", True),
        ("patch_cls_074_img_0021_64698_coord_3_4.png", True),
        ("patch_cls_175_img_0085_170981_coord_2_2.png", True),
        ("patch_cls_143_img_0013_145553_coord_2_3.png", True),
        ("patch_cls_177_img_0112_174594_coord_2_2.png", True),
        ("patch_cls_152_img_0039_156397_coord_2_4.png", True),
        ("patch_cls_068_img_0049_57891_coord_4_2.png", True),
        ("patch_cls_123_img_0059_796569_coord_2_3.png", True),
        ("patch_cls_143_img_0123_145774_coord_3_2.png", True), # ambiguous
        ("patch_cls_174_img_0054_169175_coord_3_3.png", True),
        ("patch_cls_051_img_0072_34497_coord_2_3.png", True),
        ("patch_cls_052_img_0096_35579_coord_3_3.png", True), # ambiguous
        ("patch_cls_055_img_0133_37976_coord_2_3.png", True),
        ("patch_cls_172_img_0092_167457_coord_4_3.png", True),
        ("patch_cls_068_img_0110_57851_coord_4_2.png", True),
        ("patch_cls_068_img_0034_58148_coord_2_2.png", True),
        ("patch_cls_145_img_0052_150695_coord_2_2.png", True),
        ("patch_cls_127_img_0052_118583_coord_3_2.png", True),
        ("patch_cls_049_img_0041_33396_coord_2_4.png", True),
        ("patch_cls_019_img_0074_19601_coord_4_2.png", True), #ambiguous
        ("patch_cls_190_img_0055_182520_coord_2_4.png", True),
        ("patch_cls_008_img_0029_797520_coord_3_3.png", True),
        ("patch_cls_177_img_0079_173899_coord_2_4.png", True), # beak
        ("patch_cls_113_img_0034_794589_coord_3_4.png", True),
        ("patch_cls_051_img_0072_34497_coord_3_4.png", True),
        ("patch_cls_123_img_0027_796543_coord_3_4.png", True),
        ("patch_cls_165_img_0069_163921_coord_3_3.png", True),
        ("patch_cls_117_img_0087_110946_coord_3_3.png", True),
        ("patch_cls_102_img_0040_795051_coord_3_2.png", True),
        ("patch_cls_117_img_0106_797247_coord_2_2.png", True),
        ("patch_cls_167_img_0003_164915_coord_3_3.png", True),
        ("patch_cls_146_img_0069_151229_coord_4_3.png", True),
        ("patch_cls_114_img_0091_107346_coord_2_3.png", True),
        ("patch_cls_190_img_0027_794713_coord_4_2.png", True),
        ("patch_cls_046_img_0076_31639_coord_4_3.png", True),
        ("patch_cls_173_img_0057_168424_coord_2_2.png", True), # ambiguous; tail
        ("patch_cls_127_img_0052_118583_coord_2_3.png", True),
        ("patch_cls_089_img_0090_796774_coord_2_3.png", True),
        ("patch_cls_048_img_0014_794672_coord_3_4.png", True),
        ("patch_cls_170_img_0012_166515_coord_4_3.png", True),
        ("patch_cls_175_img_0007_171523_coord_4_3.png", True), #water background
        ("patch_cls_113_img_0048_787323_coord_3_3.png", True),
        ("patch_cls_182_img_0087_176591_coord_2_3.png", True),
        ("patch_cls_050_img_0067_34416_coord_3_2.png", True), # ambiguous; beak
        ("patch_cls_059_img_0008_40942_coord_2_4.png", True), # water background
        ("patch_cls_048_img_0032_794679_coord_4_2.png", True),
        ("patch_cls_194_img_0078_185899_coord_3_2.png", True),
        ("patch_cls_010_img_0091_4096_coord_3_3.png", True),
        ("patch_cls_032_img_0012_26382_coord_4_2.png", True), # ambiguous
        ("patch_cls_046_img_0017_30979_coord_3_4.png", True), # water background
        ("patch_cls_145_img_0052_150695_coord_3_3.png", True),
        ("patch_cls_008_img_0005_2111_coord_2_2.png", True),
        ("patch_cls_034_img_0043_26990_coord_4_3.png", True),
        ("patch_cls_157_img_0052_794973_coord_3_2.png", True),
        ("patch_cls_074_img_0104_64885_coord_3_4.png", True), # ambiguous
        ("patch_cls_091_img_0056_81986_coord_2_3.png", True),
        ("patch_cls_124_img_0068_795180_coord_2_4.png", True), # ambiguous; beak
        ("patch_cls_127_img_0033_118871_coord_3_4.png", True),
        ("patch_cls_093_img_0060_84862_coord_4_3.png", True),
        ("patch_cls_156_img_0040_159101_coord_3_3.png", True),
        ("patch_cls_194_img_0098_185637_coord_3_2.png", True),
        ("patch_cls_182_img_0102_176821_coord_2_4.png", True),
        ("patch_cls_175_img_0126_171282_coord_3_3.png", True),
    ]
    # To be fixed
    # patch_cls_153_img_0043_794792_coord_0_0.png should be False

    for neuron in mturk_neurons:
        if gt_concept_label[neuron] != fbs[neuron]:
            # print(f"Neuron #{neuron} --- GT: {gt_concept_label[neuron]}, Labeled: {fbs[neuron]}")  # true: "yes", false: "no"
            neuron_imgs = neuron_to_img_dict[neuron]
            for neuron_img in neuron_imgs:
                if (img_responses[neuron_img] != gt_concept_label[neuron]): wrong_imgs.append((neuron_img, fbs[neuron]))

            num_diff += 1
    print("========================================================")
    wrong_count = 0
    for img in list(set(wrong_imgs)):
        if img not in allowed_wrong_imgs: 
            # if img[1] and (int(img[0].split("_")[-2]) in [0,1,5,6] or int(img[0].split("_")[-1].strip(".png")) in [0,1,5,6]): 
            if img[1]: 
                print(f"{img[0]} with human label {img[1]}")
                wrong_count += 1
    print(f"{wrong_count} wrong images:")
    print("========================================================")
    print(f"{num_diff}/{len(gt_label)} is incorrectly labeled")


def generate_neuron_to_img_file():
    expl_dir = "./../../expl/waterbirds"
    coord_filename = "summarized_concept_coord_resnet50_train_1.json"
    imgfiles_filename = "summarized_concept_filename_resnet50_train_1.json"
    neuron_to_img_filename = "neuron_to_img_resnet50_train_1.json"

    coords_list, imgs_list = [], []
    with open(os.path.join(expl_dir, coord_filename), "r") as f: coords_list = json.load(f)
    with open(os.path.join(expl_dir, imgfiles_filename), "r") as f: imgs_list = json.load(f)

    neuron_to_img_list = []
    for neuron_idx in range(2048):
        coords = coords_list[neuron_idx]
        imgs = imgs_list[neuron_idx]
        neuron_imgs = []
        for coord, img in zip(coords, imgs):
            cls_num = img.split(".")[0]
            img_num = "_".join(img.rstrip(".jpg").split("_")[-2:])
            coord_num = f"{int(float(coord[0]))}_{int(float(coord[1]))}"
            img_name = f"patch_cls_{cls_num}_img_{img_num}_coord_{coord_num}.png"
            neuron_imgs.append(img_name)
        neuron_to_img_list.append(neuron_imgs)
    
    dumped = json.dumps(neuron_to_img_list)
    with open(os.path.join(expl_dir, neuron_to_img_filename), "w") as f: f.write(dumped)


def generate_img_to_neuron_file():
    expl_dir = "./../../expl/waterbirds"
    coord_filename = "summarized_concept_coord_resnet50_train_1.json"
    imgfiles_filename = "summarized_concept_filename_resnet50_train_1.json"
    img_to_neuron_filename = "img_to_neuron_resnet50_train_1.json"

    coords_list, imgs_list = [], []
    with open(os.path.join(expl_dir, coord_filename), "r") as f: coords_list = json.load(f)
    with open(os.path.join(expl_dir, imgfiles_filename), "r") as f: imgs_list = json.load(f)

    img_to_neuron_dict = dict()
    for neuron_idx, (coords, imgs) in enumerate(zip(coords_list, imgs_list)):
        for coord, img in zip(coords, imgs):
            cls_num = img.split(".")[0]
            img_num = "_".join(img.rstrip(".jpg").split("_")[-2:])
            coord_num = f"{int(float(coord[0]))}_{int(float(coord[1]))}"
            img_name = f"patch_cls_{cls_num}_img_{img_num}_coord_{coord_num}.png"
            if img_name in img_to_neuron_dict: img_to_neuron_dict[img_name].append(neuron_idx)
            else: img_to_neuron_dict[img_name] = [neuron_idx] 
    
    dumped = json.dumps(img_to_neuron_dict)
    with open(os.path.join(expl_dir, img_to_neuron_filename), "w") as f: f.write(dumped)


def summarize_waterbirds_concept_files():
    expl_dir = "./../../expl/waterbirds"
    coord_filename = "concept_coord_resnet50_train_1.json"
    new_coord_filename = "summarized_concept_coord_resnet50_train_1.json"
    new_coords = []
    with open(os.path.join(expl_dir, coord_filename), "r") as f:
        coords = json.load(f)
        for coord in coords:
            new_coords.append(coord[:3])
    dumped = json.dumps(new_coords)
    with open(os.path.join(expl_dir, new_coord_filename), "w") as f: f.write(dumped)

    imgfiles_filename = "concept_filename_resnet50_train_1.json"
    new_imgfiles_filename = "summarized_concept_filename_resnet50_train_1.json"
    new_imgfiles = []
    with open(os.path.join(expl_dir, imgfiles_filename), "r") as f:
        imgfile_arrs = json.load(f)
        for imgfiles in imgfile_arrs:
            new_imgfiles.append([file.lstrip("/nvmescratch/seongmin/hfb/data/waterbirds/images/") for file in imgfiles[:3]])
    dumped = json.dumps(new_imgfiles)
    with open(os.path.join(expl_dir, new_imgfiles_filename), "w") as f: f.write(dumped)

def cluster_waterbirds_neurons():
    expl_dir = "./../../expl/waterbirds"
    new_coord_filename = "summarized_concept_coord_resnet50_train_1.json"
    new_imgfiles_filename = "summarized_concept_filename_resnet50_train_1.json"
    with open(os.path.join(expl_dir, new_coord_filename), "r") as f:
        coords_arrs = json.load(f)
    with open(os.path.join(expl_dir, new_imgfiles_filename), "r") as f:
        imgfiles_arrs = json.load(f)

    assert len(coords_arrs) == len(imgfiles_arrs)
    assert len(coords_arrs) == 2048 
    num_neurons = 2048
    
    cluster = dict()
    for neuron_idx in tqdm(range(num_neurons)):
        coords = coords_arrs[neuron_idx]
        imgfiles = imgfiles_arrs[neuron_idx]
        key = []
        for coord, imgfile in zip(coords, imgfiles):
            key.append((tuple(coord), imgfile))
        key = str(key)
        if key in cluster: cluster[key].append(neuron_idx)
        else: cluster[key] = [neuron_idx]

    print(len(cluster.keys()), "neuron clusters created")
    neuron_cluster_filename = "neuron_concept_cluster_1.json"
    dumped = json.dumps(cluster)
    with open(os.path.join(expl_dir, neuron_cluster_filename), "w") as f: f.write(dumped)

def cluster_celeba_neurons():
    # TODO: Not for clustering but counting the patch numbers
    expl_dir = "./../../expl/biased_celeba"
    new_coord_filename = "concept_coord_cnn_train_2.json"
    new_imgfiles_filename = "concept_filename_cnn_train_2.json"
    with open(os.path.join(expl_dir, new_coord_filename), "r") as f:
        coords_arrs = json.load(f)
    with open(os.path.join(expl_dir, new_imgfiles_filename), "r") as f:
        imgfiles_arrs = json.load(f)

    assert len(coords_arrs) == len(imgfiles_arrs)
    assert len(coords_arrs) == 2048 
    num_neurons = 2048
    
    patches = []
    for neuron_idx in tqdm(range(num_neurons)):
        coords = coords_arrs[neuron_idx]
        imgfiles = imgfiles_arrs[neuron_idx]
        key = []
        for coord, imgfile in zip(coords[:3], imgfiles[:3]):
            patch_name = f"{imgfile.split('.')[0]}_{int(coord[0])}_{int(coord[1])}"
            patches.append(patch_name)
    patches = list(set(patches))
    print(len(patches))


if __name__ == "__main__":
    # generate_attention_mturk_csv()
    # generate_attention_mturk_result_json()
    # generate_pruning_mturk_csv()
    generate_pruning_mturk_result_json()
    # summarize_waterbirds_concept_files()
    # cluster_waterbirds_neurons()
    # generate_img_to_neuron_file()
    # generate_neuron_to_img_file()
