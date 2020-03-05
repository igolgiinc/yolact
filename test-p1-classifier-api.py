import requests
import json
import time
import pprint

post_json = {
    #"input": "http://10.1.10.190/image1000000_9.png",
    #"input": "http://10.1.10.110:8080/openoutputs/image1000000_65.png",
    #"input": "http://10.1.10.110:8080/openoutputs/image1000000_1.png",
    #"input": "http://10.1.10.110:8080/openoutputs/classifier_input/images/test_0215.jpg",
    "input": "http://10.1.10.190/classify/image1000000_10.png",
    "type": "stream",
    #"output_dir": "/mnt/bigdrive1/cnn/outputs/",
    "output_dir": "/tmp/",
    "config": {"detection_threshold": 0.50},
    "mode": "image",
}

custom_headers = {'content-type': 'application/json'}

def send_post_request(input_img=None):
    if input_img:
        #post_json["input"] = "http://10.1.10.110:8080/openoutputs/classifier_input/images/" + input_img
        post_json["input"] = "http://10.1.10.190/classify/" + input_img
        
    pprint.pprint(post_json)
    post_json_dumps = json.dumps(post_json)

    response_json_id = -1
    
    while (True):
        retval = 0
        try:
            response = requests.post('http://10.1.10.110:11000/api/v0/classify/', headers=custom_headers, data=post_json_dumps)
        except requests.exceptions.ConnectionError:
            print("HTTP to webserver failed with ConnectionError.")
            retval = -1
        except requests.exceptions.Timeout:
            print("HTTP to webserver timed out after 30 sec.")
            retval = -1
        except requests.URLRequired:
            print("Invalid URL: " + str(url_name))
            retval = -1
        except requests.TooManyRedirects:
            print("Too many redirects for: " + str(url_name))
            retval = -1
        except requests.exceptions.RequestException:
            print("Ambiguous error for URL: " + str(url_name))
            retval = -1
        else:
            print("POST RESPONSE STATUS CODE: ", response.status_code)
            if (response.status_code == requests.codes.created):
                response_json = response.json()
                response_json_id = response_json["id"]
                print("POST RESPONSE: ", response_json)
                break
            else:
                #print("Retrying POST")
                response_json = response.json()
                #response_json_id = response_json["id"]
                print("POST RESPONSE: ", response_json)                
                break
            
        if (retval == -1):
            time.sleep(1)

    return response_json_id
            
def send_get_request(response_id):
    
    get_url = 'http://10.1.10.110:11000/api/v0/classify/%d/' % (response_id,)

    while (True):
        retval = 0
        try:
            response_get = requests.get(get_url)
        except requests.exceptions.ConnectionError:
            print("HTTP to webserver failed with ConnectionError.")
            retval = -1
        except requests.exceptions.Timeout:
            print("HTTP to webserver timed out after 30 sec.")
            retval = -1
        except requests.URLRequired:
            print("Invalid URL: " + str(url_name))
            retval = -1
        except requests.TooManyRedirects:
            print("Too many redirects for: " + str(url_name))
            retval = -1
        except requests.exceptions.RequestException:
            print("Ambiguous error for URL: " + str(url_name))
            retval = -1
        else:
            print("Size in bytes for response: ", len(response_get.content))
            print("GET response status code: ", response_get.status_code)
            response_json = response_get.json()        
            print(response_json)
            if (response_json["status"] == "finished"):
                break
            
        if ((retval == -1) or ((retval == 0) and (response_json["status"] == "running"))):
            time.sleep(1)
            #break
        
    #response_results = response_json['results']
    #printresponse_json


if __name__ == "__main__":
    response_id_list = []

    print(" About to send post request")
    #response_id = send_post_request()
    #response_id_list.append(response_id)

    for i in range(10,11):
        print(" * i = ", i)
        #response_id = send_post_request("test_%04d" % i + ".jpg")
        response_id = send_post_request("image1000000_%d" % i + ".png")
        response_id_list.append(response_id)
        #time.sleep(1.0) # 50ms sleep
        if (i % 30) == 0:
            for response_id in response_id_list:
                if response_id >= 0:
                    send_get_request(response_id)
            response_id_list = []

    time.sleep(1.0)
    for response_id in response_id_list:
        if response_id >= 0:
            send_get_request(response_id)
    
