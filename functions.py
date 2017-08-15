import pandas as pd
import geopandas as gpd
import datetime
import numpy as np
from shapely.geometry import Point, LineString
import shapely.wkt
import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import psycopg2
import cv2
import scipy.ndimage as nd

def checkDB():
    connect_str = "dbname='squidbike' user='squidbike' \
    host='rds-postgresql-10mintutorial.cbz1xmmdmpva.us-east-2.rds.amazonaws.com' \
    port='5432' password='squidbikesql'"
    conn = psycopg2.connect(connect_str)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bike_trip_testingGeoff; ")
    df = pd.DataFrame(cursor.fetchall(),columns=['dbID','timestamp',
                                             'v_value', 'geometry',
                                             'image_url', 'image_lab',
                                             'trip_id', 'bikelane_id',
                                             'original_x',' original_y',
                                             'snapped_x',' snapped_y', 'defect_score',
                                             'lane_marking_score','color_score','cv_score'])
    trackIDs = df.trip_id.unique()
    trackIDs = map(int,trackIDs)
    
    return trackIDs

def wrapperFunction():
    #load google sheets keys
    google_sheet_key = '1pGzUAMBOhefHnCUw319jaEqcllw2rMYUEvsQsfpzv7Y'
    worksheet = 'Form Responses 1'
    #load trips from google sheet
    new_trip = get_user_trip(google_sheet_key,worksheet)
    #load shapefiles of NYC bikelanes
    bufers = gpd.read_file('/home/ubuntu/squid-bike/data/nyc-bike-routes/nyc_bike_30mbuffer_epsg=3857.shp')
    bikelanes = gpd.read_file('/home/ubuntu/squid-bike/data/nyc-bike-routes/nyc_bike_routes_2017.shp')
    #trips id loaded in the DB
    loadedTrips = checkDB()
    print "new trips: ",new_trip 
    print "loadedTrips: ",loadedTrips   
    for tripID in new_trip:
        #if the trip id is not in the db
        if not tripID in loadedTrips:
            #try to run the whole process:
            try:
                print 'downloading data trip id: ',tripID
                data = downloadData(tripID)
                print 'data downloaded, snaping...'
                pointsDF = snapToBikelane(bikelaneDF = bikelanes, bufersDF = bufers, pointsDF = data)
                print 'points snapped'
                #cvDF = computer_vision_score_DF(pointsDF)
                print 'computer vision score added, saving to db'
                save_into_db(cvDF)
                print 'data saved'
            except:
                print 'trip id failed:',tripID
                pass
     
    
    
    return
        
        
    
    
def getOSCjson(OSCid):
    '''
    This function takes a OSC track id and
    returns a json file load in a Python dict
    '''

    query = "curl -s 'http://openstreetcam.org/details' -H 'Referer: http://openstreetcam.org/details/" + str(OSCid) + "/0' -H 'X-Requested-With: XMLHttpRequest' -H 'Connection: keep-alive' --data 'id=" + str(OSCid) + "&platform=web' --compressed >> osc.json"
    os.system(query)

    #read json file
    with open('osc.json') as data_file:
        data = json.load(data_file)

    os.system('rm osc.json')
    
    print 'OSC json file downloaded' 
    
    return data


def queryCV(pictureURL):
    query = '''curl -s 'https://southcentralus.api.cognitive.microsoft.com/customvision/v1.0/Prediction/16d2699a-3afd-4548-a7d3-2263abf5be63/url?iterationId=d6c29a8a-8001-49eb-b497-9f8b7109dfe7&application=quicktest' -H 'Origin: https://customvision.ai' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: en-US,en;q=0.8,es;q=0.6' -H 'Prediction-Key: '''+os.getenv('CVPREDICTIONKEY')+''' ' -H 'Content-Type: application/json;charset=UTF-8' -H 'Training-Key: '''+os.getenv('CVTRAININGKEY')+''' ' -H 'Accept: application/json, text/plain, */*' -H 'Referer: https://customvision.ai/projects/16d2699a-3afd-4548-a7d3-2263abf5be63' -H 'Connection: keep-alive' -H 'DNT: 1' --data-binary '{"Url":"'''+pictureURL+'''"}' --compressed >> photoLabel.json'''
    os.system(query)
    with open('photoLabel.json') as data_file:
        #print data_file
        data = json.load(data_file)
    os.system('rm photoLabel.json')

    highest = 0
    label = ''
    try:
        for i in range(len(data['Predictions'])):
            prob = data['Predictions'][i]['Probability']
            if  prob > highest:
                highest = prob
                label = data['Predictions'][i]['Tag']
            else:
                pass
    except KeyError:
        label = 'Error'

    return label








def downloadData(OSCid, X = True, Y = True, Z = True):
    '''
    This function takes a text file from OSC in the phone
    The axis we want to consider to compute the final vector (X, Y, Z)
    And the output formats and file
    and returns a dataframe with the V values for each gps coordinate point
    '''


    apiOutput = getOSCjson(OSCid)
    apiOutput = apiOutput['osv']
    textfile='http://openstreetcam.org/'+apiOutput['meta_data_filename']

    
    #read original data from file within track.txt.gz used by OSC to store sensor data
    data = pd.read_csv(textfile,sep=';',
                   skiprows=[0],
                   skipfooter=1,
                   usecols=[0,1,2,9,10,11,15], #14 or 15 identify video or picture
                   header=None,
                   engine = 'python') 
    
    #naming columns
    #timestam, coordinates, accelerometer data
    #point id of the photo
    data.columns = ['timestamp','long','lat','accelerationX','accelerationY','accelerationZ','point_id']


    #remove rows that has all columns (except timestamp) empty
    emtpy = data.iloc[:,1:].isnull().sum(axis=1) == data.shape[1]-1
    data = data.loc[~emtpy,:]
    data.index=range(data.shape[0])

    #create a photos data framse using only rows that have a point_id    
    dataPhotos = data.loc[~data.point_id.isnull(),['timestamp','point_id']]
    data.drop(['point_id'],axis=1,inplace=True)


    #Create accelerometer GeoDataframe
    gpsDataPoints =  data.loc[~ (data['long'].isnull()),['timestamp','long','lat']]
    
    #set an aribatry index for points with accelerometer data based on the original index
    gpsDataPoints['pointIndex'] = gpsDataPoints.index
    
    
    geometry = []
    for i in range(len(gpsDataPoints.index)):
        if i == (len(gpsDataPoints.index)-1):
            line = np.nan
        else:
            #get start and end points for each line
            startPoint = Point(gpsDataPoints['long'].loc[gpsDataPoints.index[i]], gpsDataPoints['lat'].loc[gpsDataPoints.index[i]])
            endPoint = Point(gpsDataPoints['long'].loc[gpsDataPoints.index[i+1]], gpsDataPoints['lat'].loc[gpsDataPoints.index[i+1]])
            #convert to shapely wkt
            line = LineString([startPoint,endPoint]).wkt
            geometry.append(shapely.wkt.loads(line).centroid)
    
    #remove last point (the ending point of the last line)
    gpsDataPoints = gpsDataPoints.iloc[:-1]
    
    #create GeoDataFrame with linestrings as geometry
    crs = {'init': 'epsg:4326'}
    gpsDataPoints = gpd.GeoDataFrame(gpsDataPoints, crs=crs, geometry=geometry)

    #Merge original accelerometer data with the geometry by lat and long 
    data.drop(['timestamp'],axis=1,inplace=True)
    data = data.merge(gpsDataPoints.drop(['timestamp','geometry'],axis=1),how='left')
    
    #every accelerometer reading gets assign to the point 'behind'
    data['pointIndex'] = data['pointIndex'].fillna(method='ffill')

    #shift data lag 1 to take the vector of the difference in axis XYZ
    dataShifted = data.shift(1)
    dataShifted.drop(['long','lat','pointIndex'],axis=1,inplace=True)
    dataShifted.columns = ['accelerationXShift','accelerationYShift','accelerationZShift']

    #concatenate datasets
    data = pd.concat([data,dataShifted],axis=1)
    data.drop(['long','lat'],axis=1,inplace=True)
    #remove rows 'before' the first gps coordinate
    data.dropna(axis=0,how='any',inplace=True)

    #compute XYZ vector
    data['V'] = np.sqrt((data.accelerationX-data.accelerationXShift) ** 2 * X + \
    (data.accelerationY-data.accelerationYShift) ** 2 * Y + \
    (data.accelerationZ-data.accelerationZShift) ** 2 * Z)

    #get the sum of every lag BY line defined by the starting point (with GPS data)
    vectorInformation = data.loc[:,['pointIndex','V']].groupby(by=['pointIndex']).sum()
    vectorInformation.reset_index(inplace=True)
    
    #merge final XYZ vector information with geometry data
    gpsDataPoints = gpsDataPoints.merge(vectorInformation)


    #Create photos dataframe
    photoPoints = []
    photoV = []
    pictureUrl = []
    defect_scores = []
    lane_marking_scores = []
    color_scores = []
    cvLabels = []
    dates = []

    
    #for each photo point
    for i in range(dataPhotos.shape[0]):
        #we get the accelerometer points 'after' the photo point, and before the next
        
        #if the photo point is the last, we aggregate accelerometer data from every acce point 'after' that
        if i == (dataPhotos.shape[0]-1):
            between = (gpsDataPoints.timestamp > dataPhotos.timestamp.iloc[i])
            
        #else, we aggregate accelerometer data for acce points from this photo point up to the next one
        else:
            between = (gpsDataPoints.timestamp > dataPhotos.timestamp.iloc[i]) & (gpsDataPoints.timestamp < dataPhotos.timestamp.iloc[i+1])
            
        #get V and GPS data
        dataAggregated = gpsDataPoints.loc[between,:]
        photoV.append(dataAggregated.V.mean())
        
        
        #The first point of the selected data is used as GPS location for the photos
        #there is no matching gps photo to its true position, timestamps are different
        
        #ORIGINAL
        #photoPoints.append(dataAggregated.geometry.iloc[0])
        
        
        #if there is no data between photos, we add nan
        try:
            photoPoints.append(dataAggregated.geometry.iloc[0])
        except:
            photoPoints.append(np.nan)
        
        #conversion into timestamp

        dates.append(datetime.datetime.fromtimestamp(dataPhotos['timestamp'].iloc[i]))

        #get photo url and label
        pictureName = apiOutput['photos'][i]['name']

        oscURL = 'https://'+pictureName[0:8]+'.openstreetcam.org/'+pictureName[9:]

        pictureUrl.append(oscURL)
        
        #cvLabel = queryCV(oscURL)
        #cvLabel = np.nan
        
        w1 = -50
        w2 = 10
        w3 = 2       
        
        photoCrop = photoCroping(oscURL)
        defect_score = potholeScore(photoCrop)
        lane_marking_score = laneMarkingsScore(photoCrop)
        color_score = colScore(photoCrop)
        cv_score = w1*defect_score + w2*lane_marking_score + w3*color_score
        
        defect_scores.append(defect_score)
        lane_marking_scores.append(lane_marking_score)
        color_scores.append(color_score)
        cvLabels.append(cv_score)

    

    dataPhotos['v_value'] = photoV
    dataPhotos['geometry'] = photoPoints
    dataPhotos['image_url'] = pictureUrl
    #dataPhotos['image_lab'] = cvLabels
    dataPhotos['defect_score'] = defect_scores
    dataPhotos['lane_marking_score'] = lane_marking_scores
    dataPhotos['color_score'] = color_scores
    dataPhotos['cv_score'] = cvLabels
    dataPhotos['timestamp'] = dates
    dataPhotos['trip_id'] = OSCid
    
    dataPhotos = dataPhotos.dropna(axis=0)
    crs = {'init': 'epsg:4326'}

    dataPhotos = gpd.GeoDataFrame(dataPhotos, crs=crs, geometry = dataPhotos.geometry)


    return dataPhotos


def snapToBikelane(bikelaneDF,bufersDF,pointsDF):
    '''
    this function takes:
    - a point data set (accelerometer or photos)
    - a bikelane buffer geopandas
    - a bikelane geopandas
    and returns a new geopandas with :
    the bikelane ID where the point belongs to
    the bikeline in geometry
    the new point snap to line in geomtry
    '''
    #change projection for points, bike shp already in 3857
    pointsDF = pointsDF.to_crs(epsg=3857)
    bufersDF = bufersDF.to_crs(epsg=3857)
    bikelaneDF = bikelaneDF.to_crs(epsg=3857)

    #give the line of the bikelane to the bufersDF
    bufersDF['line'] = bikelaneDF.geometry

    #joint points with buffer
    joinData = gpd.sjoin(pointsDF, bufersDF, how="left", op='intersects')

    #get unique pointID TIMES MAY CHANGE IN THE NAME WARNING
    allThePoints = pointsDF.point_id.unique()

    #create empty lists where we store new data
    line = []
    bikelanesID = []

    #get point ID duplicated
    duplicates = joinData.point_id[joinData.point_id.duplicated()].unique()


    for i in range(len(allThePoints)):
        #check if the pointIndex is unique:
        if allThePoints[i] not in duplicates:
            #append line from joint to that index
            line.append(joinData.line.loc[joinData.point_id == allThePoints[i]].iloc[0])
            bikelanesID.append(joinData.ID_ORIGINA.loc[joinData.point_id == allThePoints[i]].iloc[0])

        else:
            #if not, append from the previous id
            line.append(joinData.line.loc[joinData.point_id == allThePoints[i-1]].iloc[0])
            bikelanesID.append(joinData.ID_ORIGINA.loc[joinData.point_id == allThePoints[i-1]].iloc[0])

    pointsDF['line'] = line
    pointsDF['bikelane_id'] = bikelanesID


    pointOnLine = []

    for i in range(pointsDF.shape[0]):
        try:
            newPoint = pointsDF.line.iloc[i].interpolate(pointsDF.line.iloc[i].project(pointsDF.geometry.iloc[i]))

        except AttributeError:
            newPoint = np.nan

        pointOnLine.append(newPoint)

    pointsDForig = pointsDF.loc[:,['geometry','point_id']]
    pointsDForig = pointsDForig.to_crs(epsg=4326)

    #convert original points to x and y
    pointsDF['original_x'] = pointsDForig.geometry.map(lambda coord: coord.x)
    pointsDF['original_y'] = pointsDForig.geometry.map(lambda coord: coord.y)

    pointsDF['geometry'] = pointOnLine

    #we change the coordinates for the snapped points
    pointsDFsnap = pointsDF[pointsDF.geometry.notnull()]
    pointsDFsnap = pointsDFsnap.to_crs(epsg=4326)
    
    pointsDFsnap['snapped_x'] = pointsDFsnap.geometry.map(lambda coord: coord.x)
    pointsDFsnap['snapped_y'] = pointsDFsnap.geometry.map(lambda coord: coord.y)

    return pointsDFsnap.drop(['line','point_id'],axis=1)





def get_user_trip (google_sheet_id,worksheet_name):
    
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/ubuntu/certs/client.json', scope)
    gc = gspread.authorize(credentials)
    
    #get the data from spreadsheet
    trips=gc.open_by_key(google_sheet_id)
    df=pd.DataFrame(trips.worksheet(worksheet_name).get_all_values())
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))    
    df.control=pd.to_numeric(df.control)
    
    
    #get the data you want to work on
    to_be_worked = df[df.control!=1]
    to_be_worked.reset_index(inplace=True)
    to_be_worked.drop('index', axis=1, inplace=True)
    
    #Update the data
    #trips=gc.open_by_key(google_sheet_id)
    #tmp=trips.worksheet(worksheet_name)
    
    #for i in df[df.control!=1].index: 
    #    tmp.update_cell(i+1,4,1)
    

    
    return list(pd.to_numeric(to_be_worked['What is the track-id of your submission']))


def update_google(google_sheet_id,worksheet_name):
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/ubuntu/certs/client.json', scope)
    gc = gspread.authorize(credentials)
    
    #get the data from spreadsheet
    trips=gc.open_by_key(google_sheet_id)
    df=pd.DataFrame(trips.worksheet(worksheet_name).get_all_values())
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))    
    df.control=pd.to_numeric(df.control)
    
    #update
    trips=gc.open_by_key(google_sheet_id)
    tmp=trips.worksheet(worksheet_name)
    
    lis=checkDB()
    
    for i in df[df.control!=1].index: 
        if df['What is the track-id of your submission'][i] in lis:
            tmp.update_cell(i+1,4,1)
    return
    

def create_table():
    #connecting to the database and get port, user and passw, dbname
    connect_str = "dbname='squidbike' user='squidbike' \
    host='rds-postgresql-10mintutorial.cbz1xmmdmpva.us-east-2.rds.amazonaws.com' \
    port='5432' password='squidbikesql'"
    conn = psycopg2.connect(connect_str)

    cursor = conn.cursor()
    #drop the table
    cursor.execute("DROP TABLE IF EXISTS bike_trip_testingGeoff")
    #create new ampty table
    cursor.execute("CREATE TABLE bike_trip_testingGeoff \
                   (id SERIAL PRIMARY KEY,\
                   timestamp Text,\
                   v_value numeric(10,5),\
                   geometry Geometry,\
                   image_url Text,\
                   image_lab Text,\
                   trip_id Text,\
                   bikelane_id Text,\
                   original_x Text,\
                   original_y Text,\
                   snapped_x Text,\
                   snapped_y Text,\
                   defect_score numeric(10,5),\
                   lane_marking_score numeric(10,5),\
                   color_score numeric(10,5),\
                   cv_score numeric(10,5))"
                  )

    conn.commit()

def save_into_db(df):
    
    pointsDF = df[df.geometry.notnull()]
    
    #connecting to the database and get port, user and passw, dbname
    connect_str = "dbname='squidbike' user='squidbike' \
    host='rds-postgresql-10mintutorial.cbz1xmmdmpva.us-east-2.rds.amazonaws.com' \
    port='5432' password='squidbikesql'"
    conn = psycopg2.connect(connect_str)
   
    cursor = conn.cursor()
    
    for index in range(len(pointsDF)):
        
        timestamp = pointsDF.timestamp.iloc[index]
        v_value = pointsDF.v_value.iloc[index]
        geometry = pointsDF.geometry.iloc[index]
        image_url = pointsDF.image_url.iloc[index]
        image_lab = pointsDF.image_lab.iloc[index]
        trip_id = pointsDF.trip_id.iloc[index]
        bikelane_id = pointsDF.bikelane_id.iloc[index]
        original_x = pointsDF.original_x.iloc[index]
        original_y = pointsDF.original_y.iloc[index]
        snapped_x = pointsDF.snapped_x.iloc[index]
        snapped_y = pointsDF.snapped_y.iloc[index]
        defect_score = pointsDF.defect_score.iloc[index]
        lane_marking_score = pointsDF.lane_marking_score.iloc[index]
        color_score = pointsDF.color_score.iloc[index]
        cv_score = pointsDF.cv_score.iloc[index]
        
        cursor.execute("INSERT INTO bike_trip_testingGeoff (timestamp, v_value, geometry, image_url, image_lab,\
        trip_id, bikelane_id, original_x, original_y, snapped_x, snapped_y, defect_score,\
        lane_marking_score, color_score, cv_score)\
        VALUES ('%s', '%s', ST_GeomFromText('%s', 4326), '%s',\
        '%s', '%s', '%s', '%s', '%s', '%s', '%s','%s','%s','%s','%s')"\
        %(timestamp, v_value, geometry, image_url, \
        image_lab, trip_id, bikelane_id, original_x,\
        original_y, snapped_x, snapped_y,defect_score, lane_marking_score, color_score, cv_score))
    
    conn.commit()


### cv / image processing functions
def getPhoto(photoURL):
    photoName = photoURL.split('/')[-1]
    os.chdir('\raw_pictures')
    os.system('wget ' + photoURL)
    img = nd.imread(photoName)
    #os.system('rm ' + photoName)
    return img

def photoCroping(photoURL):
    img = getPhoto(photoURL)
    #number of rows and number of col
    nrow, ncol = img.shape[:2]
    #clipping from half the picture to 90% of the picture removes
    #the street ahead and also the bike tire
    croppedPicture = img[nrow//2:(nrow-nrow//10),:,:]
    return croppedPicture

def filter_out_line_markings(img):
    '''
    goal is to filter on color to eliminate line markings
    a lot of code taken from here:
    http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    '''
    lower = np.array([0, 10, 50])
    upper = np.array([360, 100, 100])
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hls, lower, upper)
    res = cv2.bitwise_and(hls, hls, mask = mask)
    return res

def colScore(image): 
    lower = np.array([10,31,31])
    upper = np.array([115,255,255])    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image,image, mask= mask)
    green_pixels = mask > 100
    score = green_pixels.mean()
    return score

def laneMarkingsScore(affa):
    #determine bike lane markings score
    #separate 3 color layers`
    red, grn, blu = affa.transpose(2, 0, 1)

    #create and apply a threshold to get white colors
    thrs = 200
    wind = (red > thrs) & (grn > thrs) & (blu > thrs)
                                        
    #apply gaussian filter to white indexes 
    gf = nd.filters.gaussian_filter
    blurPhoto = gf(1.0 * wind, 40)
                                                            
    #the blur image bigger than some threshold
    #the threshold is the gray area that 
    #separates the white from the black
    wreg = gf(1.0 * wind, 40) > 0.16
    score = wreg.mean()
    return score

def potholeScore(image):
    #determine pothole / crack score  
            
    #blur photo
    md = nd.filters.median_filter
    md_blurPhoto = md(image, 5)
                                
    #filter out white line markings
    photo_no_white = filter_out_line_markings(md_blurPhoto.astype(np.uint8))
                                            
    edges_cv = cv2.Canny(photo_no_white, 200, 400)
    
    #blur edges
    blurred_edges = cv2.GaussianBlur(edges_cv,(3,3),0)
    
    #only want to keep cracks that are near other cracks or that have a minimum threshold
    bdilation = nd.morphology.binary_dilation
    berosion = nd.morphology.binary_erosion
    edges_2 = bdilation(berosion(blurred_edges, iterations=2), iterations=2)
    score = edges_2.mean()
    return score

def computer_vision_score_DF(df):
    df['defect_score'] = 0
    df['lane_marking_score'] = 0
    df['color_score'] = 0
    df['cv_score'] = 0
    w1 = -50
    w2 = 10
    w3 = 2
    photoCrop = 0
    for i in range(len(df)):
        photoURL = df.image_url.iloc[i]
        photoCrop = photoCroping(photoURL)
        df['defect_score'].iloc[i] = potholeScore(photoCrop)
        df['lane_marking_score'].iloc[i] = laneMarkingsScore(photoCrop)
        df['color_score'].iloc[i] = colScore(photoCrop)
        df['cv_score'].iloc[i] = w1*df['defect_score'].iloc[i] + w2*df['lane_marking_score'].iloc[i]\
                + w3*df['color_score'].iloc[i]
    return df
