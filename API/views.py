import base64
import os
import cv2
import numpy as np
from wsgiref.util import FileWrapper

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView
from rest_framework import generics

from API.compartment_counter import get_floor_plan_compartments
from API.model_prediction_post_processing import process_img
from API.models import FloorPlansCsvData
from API.serializer import RawImageSerializer
from API.system_state import floor_plan_in_mem_db, model, tmp_folder_path


class ImageUploadView(generics.ListAPIView):
    serializer_class = RawImageSerializer

    def get_queryset(self):
        return

    def post(self, request, *args, **kwargs):
        floor_plan_in_mem_db.floor_plan_names = []
        for file in request.FILES.getlist("floor_plan"):
            floor_plan_in_mem_db.floor_plan_names.append(file.name)
            save_image_to_folder(tmp_folder_path, file)
        return HttpResponse(status=200)


def save_image_to_folder(folder, file):
    default_storage.save(folder + file.name, ContentFile(file.read()))


def convert_image_to_base64(img_name):
    with open(img_name, "rb") as img_file:
        return base64.b64encode(img_file.read())


class ResultsView(generics.ListAPIView):
    def get_queryset(self):
        return

    def get(self, request, *args, **kwargs):
        floor_plans_json_data = []
        floor_plans_csv_data = FloorPlansCsvData()

        for file_name in floor_plan_in_mem_db.floor_plan_names:
            compartments = get_floor_plan_compartments(process_img(file_name, model))
            floor_plans_json_data.append(get_floor_plan_data_json(file_name, compartments))
            floor_plans_csv_data.append_special(file_name, compartments)

        floor_plans_csv_data.to_csv_file()
        return JsonResponse(floor_plans_json_data, safe=False, status=200)


def get_floor_plan_data_json(file_name, compartments):
    os.remove(tmp_folder_path + file_name)
    org_name = "original_" + file_name
    original_floor_plan_1 = convert_image_to_base64(tmp_folder_path + org_name)
    proc_name = "processed_" + file_name
    processed_floor_plan_1 = convert_image_to_base64(tmp_folder_path + proc_name)
    




    # yolo 1
    img_filepath = tmp_folder_path + org_name
    img = cv2.imread(img_filepath)
    img_height, img_width, img_channels = img.shape
    img_color = cv2.imread(img_filepath)
    img_color = cv2.resize(img_color,(img_width,img_height))
    img = cv2.resize(img,(img_width,img_height))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = 255-img_gray   #inverted to highlight solid lines in white
    gray = cv2.threshold(img_gray.copy(), 235,255, cv2.THRESH_BINARY)[1]

    # Find the connected components on the floor
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)# takes in gray image, connectivity to lookout for corners and output image data type.

    #Threshold to filterout noises
    gap_threshold = 20

    #Create a empty list to collect identified objects
    Walls_object=[]

    unique = np.unique(labels) #labels are in multitude, so we get only unique labels

    #Loop through the labels to highlight identified areas and gets stats(coordinates) f
    for i in range(0, len(unique)):
        component = labels == unique[i]  # enforce to select only unique label
        stat = stats[i]    # get the stats and centroid for that label
        centroid= centroids[i]
        if img_color[component].sum() == 0 or np.count_nonzero(component) < gap_threshold:   #filter out the undesirable noises
            color = 0
        else:
            Walls_object.append([i, component,stat, centroid])  # collect the labels and stats 
            color = np.random.randint(0, 255, size=3)
        img_color[component] = color    #hightlight label/components on image 

    string1 = base64.b64encode(cv2.imencode('.jpg', img_color)[1]).decode()


    # yolo 2
    img_filepath = tmp_folder_path + org_name
    img = cv2.imread(img_filepath)
    img_height, img_width, img_channels = img.shape
    img_color = cv2.imread(img_filepath)
    img_color = cv2.resize(img_color,(img_width,img_height))
    img = cv2.resize(img,(img_width,img_height))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(img_gray.copy(), 235,255, cv2.THRESH_BINARY)[1]

    # Find the connected components on the floor
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)# takes in gray image, connectivity to lookout for corners and output image data type.

    #Threshold to filterout noises
    gap_threshold = 20

    #Create a empty list to collect identified objects
    Walls_object=[]

    unique = np.unique(labels) #labels are in multitude, so we get only unique labels

    #Loop through the labels to highlight identified areas and gets stats(coordinates) f
    for i in range(0, len(unique)):
        component = labels == unique[i]  # enforce to select only unique label
        stat = stats[i]    # get the stats and centroid for that label
        centroid= centroids[i]
        if img_color[component].sum() == 0 or np.count_nonzero(component) < gap_threshold:   #filter out the undesirable noises
            color = 0
        else:
            Walls_object.append([i, component,stat, centroid])  # collect the labels and stats 
            color = np.random.randint(0, 255, size=3)
        img_color[component] = color    #hightlight label/components on image 

    string2 = base64.b64encode(cv2.imencode('.jpg', img_color)[1]).decode()





    os.remove(tmp_folder_path + org_name)
    os.remove(tmp_folder_path + proc_name)

    return {
        "original_floor_plan_name": org_name,
        "original_floor_plan_image": original_floor_plan_1.decode(),
        "processed_floor_plan_name": proc_name,
        "processed_floor_plan_image": processed_floor_plan_1.decode(),
        "processed_floor_plan_image_yolo1": string1,
        "processed_floor_plan_image_yolo2": string2,
        "living_room": compartments.living_room,
        "hall": compartments.hall,
        "bathroom": compartments.bathroom,
        "bedroom": compartments.bedroom,
        "closet": compartments.closet,
        "door": compartments.door
    }


class CsvView(generics.CreateAPIView):
    def get(self, request, *args, **kwargs):  # Create the HttpResponse object with the appropriate CSV header.
        document = open(tmp_folder_path + "data.csv", 'rb')
        response = HttpResponse(FileWrapper(document), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="data.csv"'
        return response


class IndexView(TemplateView):
    template_name = "index.html"
