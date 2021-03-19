import os,socket,sys,time
from urllib import request
from html.parser import HTMLParser
from queue import Queue
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

q = Queue()

default_thread=5

socket.setdefaulttimeout(10)
file_name = 'rec.html'
local_prefix = '/data/'
s3_prefix = 's3://recommendation.ap-southeast-1/sharezone_cdn/'
cdn_prefix = 'http://sprs-cdn.wshareit.com/'

def get_file_list(file_dir):
    if os.path.isdir(file_dir):
        all_files=[]
        for root,dirs,files in os.walk(file_dir):
            for file in files:
                all_files.append(os.path.join(root,file))
        return all_files
    elif os.path.isfile(file_dir):
        return [file_dir]


def get_file_size(path):
    file_list=get_file_list(path)
    all_size=0
    for file in file_list:
        try:
            size = float(os.path.getsize(file))
            size = round(size/1024/1024,4)
            all_size+=size
        except Exception as err:
            print(err)
            sys.stdout.flush()
    return round(all_size,4)


class htmlParser(HTMLParser):
    p_text = False
    f_text = False
    paths = []
    files = []

    def init(self):
        self.p_text = False
        self.f_text = False
        self.paths = []
        self.files = []

    def handle_starttag(self,tag,attr):
        if tag == 'p':
            self.p_text = True
        if tag == 'f':
            self.f_text = True
            
    def handle_endtag(self,tag):
        if tag == 'p':
            self.p_text = False
        if tag == 'f':
            self.f_text = False
            
    def handle_data(self,data):
        if self.p_text:
            data = data.replace("\"","")
            self.paths.append(data)
        if self.f_text:
            data = data.replace("\"","")
            self.files.append(data)
    def get_data(self):
        return self.paths,self.files


def download_if_not_exists_multi_thread():
    while not q.empty():
        item = q.get()
        a = item[0]
        b = item[1]
        print("Downloading: " + a)
        sys.stdout.flush()
        while True:
            try:
                t1=time.time()
                request.urlretrieve(a,b)
                t2=time.time()
                break
            except Exception as e:
                print('>>>>>>>>>>>>>>An error occurred when downloading, retry again after 1s.')
                print('>>>>>>>>>>>>>>Exception Information: ',end='')
                print(e)
                print('-----------------------------------------------------------------------')
                sys.stdout.flush()
                time.sleep(1)
                continue
        print("Download complete for: " + b)
        print("Speed is: " + str(round(get_file_size(b)/(t2-t1),2))+'M/s')
        sys.stdout.flush()


def get_all_file_dir(local_parent_path, path):
    local_path = local_parent_path
    os.system("mkdir -p %s" % local_path)
    while True:
        try:
            response = request.urlopen(path + file_name)
            break
        except Exception as e:
            print('>>>>>>>>>>>>>>An error occurred when getting file list, retry again after 1s.')
            print('>>>>>>>>>>>>>>Exception Information: ',end='')
            print(e)
            print('-----------------------------------------------------------------------------')
            sys.stdout.flush()
            time.sleep(1)
            continue
    html = response.read().decode("utf-8")
    parser = htmlParser()
    parser.init()
    parser.feed(html)
    paths, files = parser.get_data()
    for f in files:
        q.put((path + f,local_path + f))
    for p in paths:
        get_all_file_dir(local_path + p, path + p)



if __name__ == '__main__':
    path = sys.argv[1]
    if path[-1] != "/":
        path = path + "/"
    path = path.replace(s3_prefix, cdn_prefix)
    local_path = local_prefix+path.replace(cdn_prefix,"")
    print(path)
    sys.stdout.flush()
    get_all_file_dir(local_path, path)
    
    gt1=time.time()
    t_pool = ThreadPoolExecutor()
    for j in range(default_thread):
        t_pool.submit(download_if_not_exists_multi_thread)
        time.sleep(1)
    t_pool.shutdown()
    gt2=time.time()
    
    print('>>>>>>>>>>>>>>All Done')
    print("General Speed is: " + str(round(get_file_size(local_path)/(gt2-gt1),2))+'M/s')
    sys.stdout.flush()
