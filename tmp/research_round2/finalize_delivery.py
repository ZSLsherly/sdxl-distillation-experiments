from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
workspace = root.parent.parent
pdf = workspace / 'output/pdf/图像视频推理加速_选题打磨与压力测试_第二版_2026-10-15.pdf'
manifest_path = root / 'layout_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['physical_page_count'] == manifest['logical_page_count'] == 20
assert not manifest['overflow_logical_pages']
assert not manifest['missing_font_glyphs']
for key in ['page_bound_violations', 'content_frame_violations', 'replacement_characters']:
    assert not manifest['text_qa'][key], key
manifest['visual_qa_status'] = 'passed'
manifest['visual_qa_evidence'] = 'All 20 pages inspected in grouped montages; full pages 5,6,7,8,10,11,12,14,15 inspected; final corrected pages 5,10,14 rechecked. See visual_review.txt.'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

output_dir = workspace / 'output/research'
output_dir.mkdir(parents=True, exist_ok=True)
archive = output_dir / '第二轮选题_复核材料_2026-10-15.zip'
excluded = {'author_round2.py', 'build_report.py', 'finalize_delivery.py', 'report_content.json', 'layout_manifest.json'}
files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.json','.csv','.png','.txt'} and p.name not in excluded)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in files:
        z.write(p, 'research_round2/' + p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
delivery = {'report': str(pdf), 'report_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'pages':20,
            'materials': str(archive), 'materials_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'materials_file_count': len(files), 'gpu_experiments_run': False}
(output_dir/'delivery_manifest.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pages':20,'materials_files':len(files),'pdf_bytes':pdf.stat().st_size,'zip_bytes':archive.stat().st_size}))
