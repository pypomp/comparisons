"""Prepare a self-contained reproduction bundle; release only after independent reviews.
The delivery manifest is outside the tar to avoid circular archive hashing.
"""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'work/competitor-estimation/tfp';OUT=ROOT/'outputs/competitor_estimation/tfp/formal_v2_final';HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
 gate=read(OUT/'delivery_manifest.json');s=read(OUT/'summary.json');assert gate['numeric_audit_passed'] and len(s['rows'])==36 and sha(OUT/'summary.json')==gate['summary_sha256']
 for name,digest in gate['figure_sha256'].items():assert sha(OUT/name)==digest
 source=BASE/'formal_v2';manifest=read(source/'manifest.json')
 for name,digest in manifest['sha256'].items():
  assert sha(source/name)==digest;dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
 shutil.copyfile(source/'manifest.json',OUT/'source/manifest.json')
 reference=ROOT/'work/cuthbert-dacca-paper/source/pypomp';reference_hashes=read(source/'reference_source_hashes.json')['sha256']
 for name,digest in reference_hashes.items():
  assert sha(reference/name)==digest;dest=OUT/'reference_source/pypomp'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(reference/name,dest)
 for name,digest in read(source/'source_versions.json')['source_files'].items():
  original=BASE/'source_reference'/name;assert sha(original)==digest;dest=OUT/'reference_source/tensorflow_probability/python/experimental/mcmc'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
 for method,job in [('IF2',64995117),('IFAD',64995118),('ADPF',64995119)]:
  folder=BASE/f'formal_{method}_audit_{job}';dest=OUT/'evidence'/method;dest.mkdir(parents=True,exist_ok=True)
  assert sha(folder/'audit.json')==s['method_evidence'][method]['audit_sha256']
  for name in ['audit.json','endpoints.csv','audit_finished.py','collect_finished.py','report_CN.txt','root_review.json',f'{method}_completed_raw.tar.gz']:
   if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
 for name in ['verify_archive.py','aggregate.py','plot_report.py','package.py']:shutil.copyfile(HERE/name,OUT/name)
 (OUT/'README.txt').write_text('TFP original Dacca, three methods,36 endpoints.\nRead report_CN.txt and summary.json for results and limits.\nExtract reproduction.tar.gz and run python verify_archive.py (Python+NumPy).\nThe verification script reads saved records; it never reruns fitting or PF.\nFrozen sources and inputs are under source/. Original complete method archives and audit scripts are under evidence/.\nOriginal package/runtime versions are pinned in source/source_versions.json and source/runtime_versions.json.\nNo original terminal manifests exist; after-completion retrieval hashes are retained.\nAll99 original PyPOMP reference source/data files are under reference_source/pypomp, checked against source/reference_source_hashes.json.\nThe3 executed TFP native files are under reference_source/tensorflow_probability/python/experimental/mcmc, checked against source/source_versions.json.\nRoot and visual review records, when present, are included unchanged.\nFinal delivery_manifest.json is outside the tar to avoid circular SHA hashing.\n')
 visual=False;root=False
 if (OUT/'visual_check.json').exists():
  v=read(OUT/'visual_check.json');visual=bool(v.get('passed')) and v.get('figure_sha256')==gate['figure_sha256']
 if (OUT/'root_review.json').exists():
  r=read(OUT/'root_review.json');root=bool(r.get('passed')) and r.get('summary_sha256')==sha(OUT/'summary.json') and r.get('method_audit_sha256')=={m:v['audit_sha256'] for m,v in s['method_evidence'].items()}
 if visual and root:
  report=OUT/'report_CN.txt';text=report.read_text();text=text.replace('本候选报告已汇总36个真实终点并完成数值审核；发布状态以delivery_manifest.json为准，独立复核和图像检查分别记录。','本报告汇总36个真实终点，数值审核、独立复核与图像检查均已通过；发布状态与文件SHA见delivery_manifest.json。');report.write_text(text)
 gate['report_sha256']=sha(OUT/'report_CN.txt')
 gate.update(visual_review_passed=visual,root_review_passed=root,release_ready=bool(gate['numeric_audit_passed'] and visual and root));gate['pending']=[] if gate['release_ready'] else [x for ok,x in [(visual,'Visual review'),(root,'Independent root review')] if not ok]
 files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json','reproduction.tar.gz'])
 with tarfile.open(OUT/'reproduction.tar.gz','w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
 gate['sha256']={str(p.relative_to(OUT)):sha(p) for p in files+[OUT/'reproduction.tar.gz']}
 (OUT/'delivery_manifest.json').write_text(json.dumps(gate,indent=2)+'\n')
 print(json.dumps({'release_ready':gate['release_ready'],'numeric_audit_passed':gate['numeric_audit_passed'],'visual_review_passed':visual,'root_review_passed':root,'archive_bytes':(OUT/'reproduction.tar.gz').stat().st_size},indent=2))
if __name__=='__main__':main()
