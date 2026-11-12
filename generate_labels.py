from pathlib import Path
import json,cv2,numpy as np,pyarrow.parquet as pq,time
import argparse
parser = argparse.ArgumentParser(description="Generate pink-object wrist-camera weak labels from LeRobot exports")
parser.add_argument("dataset_spec", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=False)
spec = json.loads(args.dataset_spec.read_text())
report, allrows = [], []
for ds in spec['datasets']:
 root=Path(ds['path']);info=json.loads((root/'meta/info.json').read_text());manifest=json.loads((root/'export_manifest.json').read_text())
 for ep in manifest['episodes']:
  idx=ep['episode_index'];fmt=dict(episode_chunk=idx//info['chunks_size'],episode_index=idx)
  tab=pq.read_table(root/info['data_path'].format(**fmt));act=np.asarray(tab['action'].to_pylist());n=len(act);close=np.where((act[:-1,7]>.5)&(act[1:,7]<.25))[0]+1
  rec=dict(dataset=str(root),episode=idx,frames=n,close_candidates=close.tolist(),accepted=0)
  if not len(close):report.append(rec);continue
  c=int(close[0]);lo=max(0,c-60);hi=max(lo,c-6)
  cap=cv2.VideoCapture(str(root/info['video_path'].format(video_key='observation.images.cam_left_wrist',**fmt)));cap.set(cv2.CAP_PROP_POS_FRAMES,lo);proposals=[]
  for f in range(lo,hi):
   ok,bgr=cap.read()
   if not ok:break
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(float);rr,gg,bb=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
   mask=((rr>150)&(rr>1.35*gg)&(bb>1.10*gg)).astype('uint8');mask[:100]=0;mask[430:]=0;mask[:,:160]=0;mask[:,500:]=0
   count,ids,stats,centers=cv2.connectedComponentsWithStats(mask)
   choices=[k for k in range(1,count) if 35<=stats[k,cv2.CC_STAT_AREA]<=4500 and 5<stats[k,cv2.CC_STAT_WIDTH]<110 and 5<stats[k,cv2.CC_STAT_HEIGHT]<110]
   if len(choices)!=1:continue
   k=choices[0];uv=centers[k];proposals.append((f,uv,bgr.copy(),int(stats[k,cv2.CC_STAT_AREA])))
  cap.release()
  # Require consecutive observations and bounded movement; ambiguous tracks are masked.
  keep=[]
  for j,(f,uv,bgr,area) in enumerate(proposals):
   neighbors=[proposals[k] for k in [j-1,j+1] if 0<=k<len(proposals) and abs(proposals[k][0]-f)==1 and np.linalg.norm(proposals[k][1]-uv)<12]
   if len(neighbors)<1:continue
   keep.append((f,uv,bgr,area));allrows.append(dict(dataset=str(root),episode_index=idx,frame_index=f,uv=uv.tolist(),camera='cam_left_wrist',valid=True,weight=.5,source='first_close_prephase_unique_color_component_temporal_gate',semantics='weak_object_surface_proxy',close_frame=c))
  rec['accepted']=len(keep)
  if keep:
   f,uv,bgr,area=keep[len(keep)//2];u,v=map(int,uv);cv2.circle(bgr,(u,v),9,(0,255,255),2);name=f'{root.name}_{idx:03d}_{f:04d}.jpg';cv2.imwrite(str(out/name),bgr);rec['preview']=name
  report.append(rec)
  (out/'generation_progress.json').write_text(json.dumps({'episodes_done':len(report),'labels':len(allrows)}))
(out/'labels_proposals.json').write_text(json.dumps(allrows));(out/'generation_report.json').write_text(json.dumps({'episodes':report,'total_labels':len(allrows),'episodes_with_labels':sum(x['accepted']>0 for x in report),'status':'generated_pending_agent_visual_audit','automatic_method_scope':'pink-object color proxy; not a general object detector'},indent=2));print('DONE',len(report),len(allrows),flush=True)
