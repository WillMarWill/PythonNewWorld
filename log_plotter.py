from parse import *
from boat_parser import gps_parser, gps_parser2, int2float64, to_2326
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

#INPUT_FILE = 'test-error.log'
INPUT_FILE = 'boat-client-error.log'
#INPUT_FILE = 'monitoring_points.log'
OUTPUT_FILE = 'gps_points.log'
gps_pattern = "['0xac', '0x1', '0xd'"
gps_pattern2 = "'0xac', '0x3', '0x6'"
starting_time = datetime(2021, 6, 25, 12, 40)
sampling = 20


def Protocol_CRC8(data, size):
    i = 0;
    j = 0;
    crc = 0;
    for i in range(0,size):
        crc = crc ^ data[i];
        for j in range(0,8):
            if ((crc & 0x01) != 0):
                crc = (crc >> 1) ^ 0x8C;
            else:
                crc >>= 1;
    return crc


def Protocol_UnPack(data):
    PACKET_START=0xAC
    PACKET_END=0xAD
    PACKET_ESCAPE=0xAE
    PACKET_ESCAPE_MASK=0x80
    index = 0;
    pack_len = 0;
    crc = 0;
    buffer=[]
    extra=0

    if(len(data)<=5):
        return []
    while(1):
        if(index+extra>=len(data)):
            break;
        if ((PACKET_ESCAPE != data[index+extra-1]) and (PACKET_END == data[index+extra])):
            break;
        if ((PACKET_START == data[index+extra]) or (PACKET_END == data[index+extra])):
            index+=1
            continue
        if (PACKET_ESCAPE == data[index+extra]):
            extra+=1
            buffer += ([data[index+extra] ^ PACKET_ESCAPE_MASK]);
            pack_len+=1;
            index+=1
            continue
        buffer += [data[index+extra]]
        pack_len+=1
        index+=1
    if (pack_len > 0):
        crc = Protocol_CRC8(buffer, pack_len - 1);
        if (crc != buffer[pack_len - 1]):
            return [];  #CRC error
    if len(data)-1==index+extra:
        data=[]
    else:
        if(len(data)>index+extra):
            data=data[index+extra+1:]
        else:
            data=[]
    return buffer[:len(buffer)-1]


with open(OUTPUT_FILE, 'w') as output_file:
    with open(INPUT_FILE) as input_file:
        line = input_file.readline()
        last_line = ''
        fmt = '%Y-%m-%d %H:%M:%S,%f'
        while line:
            #print(line)
            
            if line.find(gps_pattern) != -1:
                parsed_line = parse('[{time}] [{payload}]\n', line)
                print(parsed_line)
                try:
                    if parsed_line['payload'] != last_line:
                        if datetime.strptime(parsed_line['time'], fmt) > starting_time:
                            output_file.write(last_line+'\n')
                            last_line = parsed_line['payload']
                except:
                    print(last_line)
                    
                    
            #print(line)       
            if line.find(gps_pattern2) != -1:
                parsed_line = parse('[{time}] [{payload}]\n', line)
                #print(parsed_line)
                try:
                    if parsed_line['payload'] != last_line:
                        if datetime.strptime(parsed_line['time'], fmt) > starting_time:
                            output_file.write(last_line+'\n')
                            last_line = parsed_line['payload']
                except:
                    print(last_line)
            
            line = input_file.readline()
            


with open(OUTPUT_FILE) as output_file:
    pts = []
    counter = 0
    line = output_file.readline()
#    import pdb; pdb.set_trace()
    
    while line:
        try:
            parsed_line = line.replace("\n",'').replace("'",'').split(", ")[0:]
            #print(parsed_line)
            #parsed_line = [int(x, 16) for x in r]
               
            if r == '0xac':
                pts.append(r)
                
            r = [int(x, 16) for x in r]
            print(r)
            r = Protocol_UnPack(r)
            print(r)
            
            for element in r:
                if element == '172':
                    pts.append(element)
            
            #parsed_line = [int(r[0].split('0x')[1],16) for r in findall("'{}'", line)]
            print(parsed_line)
            array = [['1']]
            start = 0
            for count, value in enumerate(parsed_line):
                if value == '0xac':

                    array.append(parsed_line[start:count])
                    start = count
            #print(array)
            
            #parsed_line = Protocol_UnPack(parsed_line)
            #print(parsed_line)
                    
            '''
            if len(parsed_line) == 19:`
                if parsed_line[3] == 0x40 and parsed_line[11] == 0x40:
                    point = gps_parser(parsed_line)
                    if point:
                        pts.append(point)
                        '''
            
            #if len(parsed_line) == 100:
                #if parsed_line[13] == 0x40 and parsed_line[21] == 0x40:
            #print(array)
            
            for r in parsed_line:
                if r and len(r)>3:
                    
                    #point = gps_parser(r)
                    #pts.append(point)
                    #print(r[0])
                    if r[1] == '0x3' and r[2] == '0x6':
                        print(r)
            for r in array:


                if r and len(r)>3:
                    
                    #point = gps_parser(r)
                    #pts.append(point)
                    #print(r[0])
                    if r[1] == '0x3' and r[2] == '0x6':
                        #print(r)
                        print(r)
                        #print(r[14:22])
                        #print(r[22:30])
                        #latitude = int2float64(r[14:22])
                        #longitude = int2float64(r[22:30])
                        #print("(%0.6f, %0.6f)"%(longitude,latitude))
                        #print(data)
                        #pts.append(data)

            #point = gps_parser2(array)
            #if point:
                #pts.append(point)
        except:
            pass
        for i in range(1,sampling):
            line = output_file.readline()
        counter +=1
        #print(counter)


route = gpd.GeoDataFrame({'geometry':pts})
df = gpd.read_file("plover.geojson")
df_hk = df.to_crs('epsg:2326')
fig, ax = plt.subplots (figsize = (10,10))
df_hk.plot(ax=ax, markersize=0.1, figsize=(20,20), legend=True)
route.plot(ax=ax, facecolor='red', markersize=0.1, edgecolor='red', legend=True)
plt.savefig('gps_logs_sampling_%s.jpg'%sampling)
plt.show()
