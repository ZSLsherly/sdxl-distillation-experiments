"""Measured throughput -> a bounded paired experiment budget, not GPU-spec extrapolation."""
import argparse,json,math

def plan(seconds, hours, runs, overhead_hours, batch):
    if min(seconds,hours,runs,batch)<=0 or not 0<=overhead_hours<hours:
        raise ValueError('Invalid budget')
    updates=int((hours-overhead_hours)*3600/(seconds*runs))//3*3
    return {'total_budget_hours':hours,'runs':runs,'reserve_eval_io_hours':overhead_hours,
       'measured_seconds_per_G_update':seconds,'max_G_updates_per_run':updates,
       'G_samples_per_run':updates*batch,'fake_D_updates_per_run':updates*5,
       'stage4_GAN_updates_per_run':updates//3,
       'estimated_training_hours_total':seconds*updates*runs/3600,
       'note':'Use same-host full-5:1, no-profile measured speed; reserve includes all evaluations/checkpoints. No efficacy prediction.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seconds-per-update',type=float,required=True)
    ap.add_argument('--hours',type=float,default=72);ap.add_argument('--runs',type=int,default=2)
    ap.add_argument('--overhead-hours',type=float,default=12);ap.add_argument('--batch',type=int,default=64)
    a=ap.parse_args();print(json.dumps(plan(a.seconds_per_update,a.hours,a.runs,a.overhead_hours,a.batch),indent=2))
