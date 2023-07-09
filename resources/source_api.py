from flask import Response, request, jsonify, make_response
from flask_restful import Resource
from src.utils import convert_pdf2images, convert_image_to_base64
from PIL import Image
import cv2
import numpy as np
import uuid
from src.utils import seperate_image, ocr_text, ocr_table

class PreprocessApi(Resource):
    def post(self):
        if 'file' not in request.files:
            return make_response("No file part", 400)
        file = request.files['file']
        if file.filename == '':
            return make_response("No selected file", 400)
        if file.mimetype!='application/pdf':
            return make_response("Upload file format is not correct", 400)
        response = file.read()
        image_metadata = convert_pdf2images(response)
        return make_response(image_metadata, 200)
        
class OCRApi(Resource):
    def post(self):
        if 'file' not in request.files:
            return make_response("No file part", 400)
        file = request.files['file']
        if file.filename == '':
            return make_response("No selected file", 400)
        if file.mimetype!='image/jpeg':
            return make_response("Upload file format is not correct", 400)
        response = file.read()
        img = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_COLOR)
        results = seperate_image(img)
        text_metadata = ocr_text(results['image'], results['texts'])
        tables_metadata = []
        if len(results['tables']) != 0:
            for table in results['tables']:
                base64_table = convert_image_to_base64(Image.fromarray(table['image']))
                id = uuid.uuid1()
                table_data = ocr_table(table)
                tables_metadata.append({'table_id': id, 'table_coordinate': table['table_coordinate'],
                                         'table_image': base64_table, 'table_data': table_data})
        return make_response({'metadata': {'text_metadata': text_metadata, 'table_metadata': tables_metadata}}, 200)
        
