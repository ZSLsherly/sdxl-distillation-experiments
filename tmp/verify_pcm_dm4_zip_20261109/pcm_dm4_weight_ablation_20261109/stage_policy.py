"""Small, explicit stage routing for SDXL ablations; no model/framework logic."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StagePlan:
    student: tuple
    dmd: tuple
    gan: tuple
    fake_policy: str

    def student_stage(self, update):
        return self.student[update % len(self.student)]

    def gan_updates_before(self, update):
        cycles, tail = divmod(update, len(self.student))
        return cycles * len(self.gan) + sum(s in self.gan for s in self.student[:tail])

    def critic_stage(self, fake_update, student_update):
        if self.fake_policy == 'crossdistill':
            return fake_update % 4 + 1
        if self.fake_policy == 'student_cycle':
            return self.student[fake_update % len(self.student)]
        if self.fake_policy == 'match_student':
            return self.student_stage(student_update)
        return 4  # Explicit final-only reference ablation, not the default.

    def metadata(self):
        return {'student_stages':list(self.student), 'dmd_stages':list(self.dmd),
                'gan_stages':list(self.gan), 'fake_policy':self.fake_policy}


def make_stage_plan(variant, student=(2,3,4), dmd=None, gan=None, fake_policy='crossdistill'):
    presets={'joint4':((2,3,4),(4,)), 'joint34':((2,3,4),(3,4)),
             'gan4':((2,3),(4,)), 'dmd_only':((2,3,4),())}
    if variant not in presets or fake_policy not in ('crossdistill','student_cycle','match_student','final'):
        raise ValueError('Unknown stage preset or fake policy')
    student=tuple(student)
    if not student or len(set(student))!=len(student) or not set(student)<={2,3,4}:
        raise ValueError('Student stages must be unique choices from 2,3,4; stage1 stays with PCM')
    default_dmd,default_gan=presets[variant]
    dmd=tuple(s for s in default_dmd if s in student) if dmd is None else tuple(dmd)
    gan=tuple(s for s in default_gan if s in student) if gan is None else tuple(gan)
    for values in (dmd,gan):
        if len(set(values))!=len(values) or not set(values)<=set(student):
            raise ValueError('Loss stages must be unique and belong to the student cycle')
    if set(dmd)|set(gan)!=set(student):
        raise ValueError('Every student stage needs DMD, GAN, or both')
    return StagePlan(student,dmd,gan,fake_policy)
