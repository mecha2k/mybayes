from __future__ import print_function

import csv
import json
import numpy
import os
import sys
import redis
import urllib.request as urllib2
from datetime import datetime, time
from time import sleep


class Redis(object):
    host = "dory.redistogo.com"
    port = 10534

    def __init__(self):
        try:
            password = os.environ["REDIS_AUTH"]
        except KeyError:
            print("Environment variable REDIS_AUTH is not set.")
            sys.exit()
        self.r = redis.StrictRedis(host=self.host, port=self.port, password=password, db=0)

    def WriteTrainSpotting(self, timestamp, tripid, seconds, live=True):
        dt = datetime.fromtimestamp(timestamp)
        day = dt.date().isoformat()
        print(dt, tripid, seconds, timestamp)

        if live:
            self.r.sadd("days", day)
            self.r.sadd(day, tripid)
            self.r.zadd(tripid, seconds, timestamp)

    def FindArrivals(self, start_hour=16, end_hour=18):
        days = self.r.smembers("days")
        print(days)

        start_time = time(hour=start_hour)
        end_time = time(hour=end_hour)
        arrival_map = {}
        for day in days:
            tripids = self.r.smembers(day)
            for tripid in tripids:
                pred_dt = self.GetPredictedArrival(tripid)
                pred_time = pred_dt.time()
                if start_time < pred_time < end_time:
                    arrival_map.setdefault(day, []).append(pred_dt)

        return arrival_map

    def GetPredictedArrival(self, tripid):
        pair = self.r.zrange(tripid, 0, 1, withscores=True)
        timestamp, seconds = pair[0]
        pred_ts = float(timestamp) + seconds
        pred_dt = datetime.fromtimestamp(pred_ts)
        return pred_dt


class TrainSpotting(object):
    def __init__(self, t):
        self.timestamp = int(t[0])
        self.tripid = t[2]
        self.seconds = int(t[6])


def ReadCsv(url="http://developer.mbta.com/lib/rthr/red.csv"):
    fp = urllib2.urlopen(url)
    reader = csv.reader(fp)
    tss = []
    for t in reader:
        if t[5] != "Kendall/MIT":
            continue
        if t[3] != "Braintree":
            continue
        ts = TrainSpotting(t)
        tss.append(ts)
    fp.close()
    return tss


def ReadJson():
    url = "http://developer.mbta.com/lib/rthr/red.json"
    json_text = urllib2.urlopen(url).read()
    json_obj = json.loads(json_text)
    print(json_obj)


def ReadAndStore(red):
    tss = ReadCsv()
    for ts in tss:
        red.WriteTrainSpotting(ts.timestamp, ts.tripid, ts.seconds)


def Loop(red, start_time, end_time, delay=60):
    if datetime.now() < start_time:
        diff = start_time - datetime.now()
        print("Sleeping", diff)
        sleep(diff.total_seconds())
    while datetime.now() < end_time:
        print("Collecting")
        ReadAndStore(red)
        sleep(delay)


def TodayAt(hour):
    now = datetime.now()
    return datetime.combine(now, time(hour=hour))


def GetInterarrivals(arrival_map):
    interarrival_seconds = []
    for day, arrivals in sorted(arrival_map.items()):
        print(day, len(arrivals))
        arrivals.sort()
        diffs = numpy.diff(arrivals)
        diffs = [diff.total_seconds() for diff in diffs]
        interarrival_seconds.extend(diffs)

    return interarrival_seconds


def main(script, command="collect"):
    red = Redis()
    if command == "collect":
        start = TodayAt(16)
        end = TodayAt(18)
        print(start, end)
        Loop(red, start, end)
    elif command == "report":
        arrival_map = red.FindArrivals()
        interarrivals = GetInterarrivals(arrival_map)
        print(repr(interarrivals))


if __name__ == "__main__":
    main(*sys.argv)