import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const dir='C:/Users/sherl/Documents/ChatGPT/推理加速/tmp/weekly_1014';
for (const [idx,file] of ['D:/study/论文集/推理加速/周报/周报0916.pptx','D:/study/论文集/推理加速/周报/周报.pptx'].entries()) {
 const p=await PresentationFile.importPptx(await FileBlob.load(file));
 await fs.writeFile(`${dir}/ref${idx}.webp`,new Uint8Array(await (await p.export({format:'webp',montage:true,scale:.5})).arrayBuffer()));
 for(const n of [0,2]) await fs.writeFile(`${dir}/ref${idx}-slide${n+1}.png`,new Uint8Array(await (await p.export({slide:p.slides.items[n],format:'png',scale:1})).arrayBuffer()));
 console.log('Reference',idx,p.slides.items.length);
}
