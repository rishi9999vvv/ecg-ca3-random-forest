import sys, json, time
from pathlib import Path
base = Path(__file__).resolve().parent

sys.path.insert(0, str(base))
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import app
summary = {'scope': 'Classroom sanity check on three 30-second MIT-BIH excerpts; not a held-out benchmark. Training overlap is unknown.', 'records': []}
all_true, all_pred = [], []
symbol_map = {**{s: 'N' for s in ['N','L','R','e','j']}, **{s:'S' for s in ['A','a','J','S']}, **{s:'V' for s in ['V','E']}, 'F':'F', '/':'Q', 'f':'Q', 'Q':'Q'}
for name in ['100','119','200']:
    values=np.loadtxt(base / f'samples/mitbih_{name}_30s.csv', skiprows=1)
    start=time.perf_counter()
    result=app.analyze_signal(values,360)
    elapsed=time.perf_counter()-start
    annotations=json.loads((base/f'samples/mitbih_{name}_annotations.json').read_text())
    eligible=[(sample, symbol_map[symbol]) for sample,symbol in zip(annotations['samples'],annotations['symbols']) if symbol in symbol_map]
    used=set(); truth=[];predicted=[]
    for beat in result['beats']:
        pos=round(beat['time_seconds']*360)
        options=[(abs(sample-pos),i,label) for i,(sample,label) in enumerate(eligible) if i not in used]
        distance,i,label=min(options)
        if distance<=54:
            used.add(i);truth.append(label);predicted.append(beat['type'])
    all_true.extend(truth);all_pred.extend(predicted)
    summary['records'].append({'record':name,'duration_seconds':30,'detected_beats':result['detected_beats'],'classified_beats':result['classified_beats'],'matched_reference_beats':len(truth),'correct_predictions':sum(a==b for a,b in zip(truth,predicted)),'sample_accuracy':sum(a==b for a,b in zip(truth,predicted))/len(truth),'heart_rate_bpm':result['heart_rate_bpm'],'inference_seconds':round(elapsed,3),'distribution':result['distribution']})
labels=['F','N','Q','S','V']
summary['label_order']=labels
summary['matching']='Greedy one-to-one nearest annotation match within 150 ms. First/last detected peaks and incomplete beat windows excluded. Includes detector errors; annotation label mapping is recorded in this script.'
summary['classification_report']=classification_report(all_true,all_pred,labels=labels,output_dict=True,zero_division=0)
summary['confusion_matrix']=confusion_matrix(all_true,all_pred,labels=labels).tolist()
(base/'reports/current_pipeline_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

