from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from typing import List, Optional
import json
import h5py
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
import pandas as pd

N_TURNS = 10

def hard_yn_reward(text):
    text = text.lower()
    return text == 'yes' or text == 'no'

def soft_yn_reward(text):
    text = text.lower()
    return 'yes' in text or 'no' in text

def conservative_yn_reward(text):
    text = text.lower()
    key_words = ['not', 'don\'t', 'can\'t', 
                 'don’t', 'can’t', 
                 'cannot', 'fairly', 
                 'could', 'think so', 
                 'okay', 'maybe', 
                 'yes', 'no', 
                 'looks', 'appears', 
                 'tell', 'mostly just']
    return any([word in text for word in key_words])

yn_reward_fs = {'none': None, 'soft': soft_yn_reward, 'hard': hard_yn_reward, 'conservative': conservative_yn_reward}

class CutoffRule(ABC):
    @abstractmethod
    def apply_rule(self, scene: Scene, event: Event) -> bool:
        pass

class PercentileCutoffRule:
    def __init__(self, goal_value: float, percentile: float):
        self.goal_value = goal_value
        self.percentile = percentile

    def apply_rule(self, scene: Scene, event: Event):
        progress = sum([ev.progress for ev in event.get_events()]) / (self.goal_value-scene.initial_val)
        return progress >= self.percentile

@dataclass
class Event:
    def append(self, ev: Event, link_forward=False):
        ev.prev = self
        if link_forward:
            self.next = ev
        ev.scene = self.scene
        return ev
    
    def get_events(self, direction="prev"):
        if direction == "prev":
            func = lambda ev: ev.prev
        elif direction == "next":
            func = lambda ev: ev.next
        else:
            raise NotImplementedError
        events = []
        ev = self
        while ev is not None:
            events.append(ev)
            ev = func(ev)
        if direction == 'prev':
            events.reverse()
        return events

    def get_all_events(self):
        return self.get_events() + self.get_events('next')[1:]
    
    def is_final(self):
        return isinstance(self, StopEvent) or (self.scene.cutoff_rule is not None and self.scene.cutoff_rule.apply_rule(self.scene, self))

@dataclass
class QuestionEvent(Event):
    question: str
    progress: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return self.question

@dataclass
class AnswerEvent(Event):
    answer: str
    reward: float
    progress: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return self.answer

@dataclass
class StopEvent(Event):
    progress: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return '<stop>'

@dataclass
class Scene:
    events: List[Event]
    initial_val: Optional[float]
    cutoff_rule: Optional[CutoffRule]
    
    @classmethod
    def from_json(cls, df, reward, progress):
        events = []
        user = df[df['name'] == 'Context'].reset_index()
        chatbot = df[df['name'] == 'Response'].reset_index()
        for i in range(len(user)):
            events.append(QuestionEvent(user.loc[i, 'line'], 0.0, None, None, None))
            events.append(AnswerEvent(chatbot.loc[i, 'line'], 
                                      reward[i] if reward is not None else 0.0, 
                                      0.0 if progress is None else progress[i+1], None, None, None))
        scene = cls(events, 0.0 if progress is None else progress[0], None)
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev = p
        for ev in events:
            ev.scene = scene
        return scene
    
    @classmethod
    def from_json_cuttoff(cls, df, progress, cutoff_rule, yn_reward, yn_reward_f):
        events = []
        user = df[df['name'] == 'Context'].reset_index()
        chatbot = df[df['name'] == 'Response'].reset_index()
        for i in range(len(user)):
            events.append(QuestionEvent(user.loc[i, 'line'], 
                                        0.0, None, None, None))
            events.append(AnswerEvent(chatbot.loc[i, 'line'], 
                                      -1.0 + (yn_reward if yn_reward_f is not None and yn_reward_f(chatbot.loc[i, 'line']) else 0.0), 
                                      0.0 if progress is None else progress[i+1], 
                                      None, None, None))
        scene = cls(events, 0.0 if progress is None else progress[0], cutoff_rule)
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev = p
        for ev in events:
            ev.scene = scene
        for i, ev in enumerate(scene.events):
            if isinstance(ev, AnswerEvent) and ev.is_final():
                scene.events = scene.events[:(i+1)]
                scene.events[-1].next = None
                ev.reward = 0.0 + (yn_reward if yn_reward_f is not None and yn_reward_f(scene.events[-1].answer) else 0.0)
        return scene
    
    @classmethod
    def from_json_stops(cls, df, reward, progress):
        scenes = []
        user = df[df['name'] == 'Context'].reset_index()
        chatbot = df[df['name'] == 'Response'].reset_index()
        for x in range(len(user)+1):
            events = []
            for i in range(x):
                events.append(QuestionEvent(user.loc[i, 'line'], 0.0, None, None, None))
                events.append(AnswerEvent(chatbot.loc[i, 'line'], 
                                          reward[i] if reward is not None else 0.0, 
                                          0.0 if progress is None else progress[i+1], None, None, None))
            if x < len(user):
                events.append(StopEvent(0.0, None, None, None))
            scene = cls(events, 0.0 if progress is None else progress[0], None)
            for p, n in zip(events[:-1], events[1:]):
                p.next = n
                n.prev = p
            for ev in events:
                ev.scene = scene
            scenes.append(scene)
        return scenes
    
class SocraticDialogueData:
    def __init__(self, data_path: str, 
                 reward_cache: Optional[str]=None, 
                 reward_shift: float=0.0, 
                 reward_scale: float=1.0, 
                 addition_scenes: Optional[List[Scene]]=None, 
                 mode: str='env_stops', 
                 cutoff_rule: Optional[CutoffRule]=None, 
                 yn_reward: float=-2.0, yn_reward_kind: str='none'):
        assert mode in ['agent_stops', 'env_stops', '10_stop']
        assert yn_reward_kind in yn_reward_fs
        if mode == 'env_stops':
            if cutoff_rule is None:
                cutoff_rule = PercentileCutoffRule(1.0, 0.5)
            # assert reward_cache is not None
        yn_reward_f = yn_reward_fs[yn_reward_kind]
        # with open(data_path, 'r') as f:
        #     data = json.load(f)
        df = pd.read_csv(data_path, sep=',')
        df['line'] = df['line'].apply(lambda x: x.strip())
        if reward_cache is not None:
            with open(reward_cache, 'r') as f:
                reward = json.load(f)
            progress = reward
            reward = [[item * reward_scale + reward_shift for item in rs[1:]] for rs in reward]
        else:
            progress = None
            reward = None
        if mode == 'agent_stops':
            self.scenes = sum([Scene.from_json_stops(df[df['id'] == i+1].reset_index(), 
                                                     reward if reward is None else reward[i], 
                                                     progress[i] if progress is not None else None) for i in range(len(df['id'].unique().tolist()))], [])
        elif mode == 'env_stops':
            # maybe make reward 0 or -1 here
            self.scenes = [Scene.from_json_cuttoff(df[df['id'] == i+1].reset_index(), 
                                                   progress[i] if progress is not None else None, 
                                                   cutoff_rule, yn_reward, yn_reward_f) for i in range(len(df['id'].unique().tolist()))]
        elif mode == '10_stop':
            self.scenes = [Scene.from_json(df[df['id'] == i+1].reset_index(), 
                                           reward if reward is None else reward[i], 
                                           progress[i] if progress is not None else None) for i in range(len(df['id'].unique().tolist()))]
        else:
            raise NotImplementedError
        if addition_scenes is not None:
            self.scenes += addition_scenes
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]

