'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const route = (p, line='') => '/?path=' + encodeURIComponent(p) + (line ? '#L'+line : '');
const raw = p => '/raw?path='+encodeURIComponent(p);
const path = new URLSearchParams(location.search).get('path') || '';
let activeFile, requestSequence=0;
async function api(endpoint, params={}) {const r=await fetch('/api/'+endpoint+'?'+new URLSearchParams(params));if(!r.ok)throw new Error('Unable to read this path');return r.json();}
function link(url, base) {
  if (/^https?:\/\//i.test(url)) return url;
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('//') || url.startsWith('/')) return '#';
  const [file, anchor=''] = url.split('#');
  const parts=(base.split('/').slice(0,-1).join('/')+'/'+file).split('/');const out=[];
  for(const part of parts){if(part==='..')out.pop();else if(part && part!=='.')out.push(part);}
  return route(file ? out.join('/') : base, /^L\d+$/.test(anchor) ? anchor.slice(1) : '');
}
function inline(text, base) {
  const held=[];const hold=html=>{held.push(html);return '\u0001'+(held.length-1)+'\u0002';};
  let s=text.replace(/`([^`]+)`/g,(_,v)=>hold('<code>'+esc(v)+'</code>'));
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,label,url)=>hold('<a href="'+esc(link(url,base))+'">'+esc(label)+'</a>'));
  s=esc(s).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  for(let i=0;i<3;i++)s=s.replace(/\u0001(\d+)\u0002/g,(_,j)=>held[j]);
  return s;
}
function markdown(text, base) {
  const lines=text.split('\n'), result=[];let i=0;
  const cells=s=>s.trim().replace(/^\||\|$/g,'').split('|').map(x=>x.trim());
  while(i<lines.length){let s=lines[i];
    if(!s.trim()){i++;continue;}
    if(/^```/.test(s)){const block=[];i++;while(i<lines.length&&!/^```/.test(lines[i]))block.push(lines[i++]);i++;result.push('<pre><code>'+esc(block.join('\n'))+'</code></pre>');continue;}
    const heading=s.match(/^(#{1,6})\s+(.+)/);if(heading){const n=heading[1].length;result.push('<h'+n+'>'+inline(heading[2],base)+'</h'+n+'>');i++;continue;}
    if(s.includes('|')&&i+1<lines.length&&/^\s*\|?\s*:?-{3}/.test(lines[i+1])){
      let html='<div class="table-scroll"><table><thead><tr>'+cells(s).map(v=>'<th>'+inline(v,base)+'</th>').join('')+'</tr></thead><tbody>';i+=2;
      while(i<lines.length&&lines[i].includes('|')){html+='<tr>'+cells(lines[i++]).map(v=>'<td>'+inline(v,base)+'</td>').join('')+'</tr>';}
      result.push(html+'</tbody></table></div>');continue;
    }
    if(/^\s*[-*]\s+/.test(s)){const list=[];while(i<lines.length&&/^\s*[-*]\s+/.test(lines[i]))list.push('<li>'+inline(lines[i++].replace(/^\s*[-*]\s+/,''),base)+'</li>');result.push('<ul>'+list.join('')+'</ul>');continue;}
    const para=[s];i++;while(i<lines.length&&lines[i].trim()&&!/^(#|```|[-*]\s|\|)/.test(lines[i]))para.push(lines[i++]);result.push('<p>'+inline(para.join('\n'),base)+'</p>');
  }return '<article class="markdown">'+result.join('\n')+'</article>';
}
function source(text, patch=false){const lines=text.split('\n');let added=0,removed=0;
  const html=lines.map((s,i)=>{let cls='';if(patch){if(s.startsWith('diff --git'))cls='file-head';else if(s.startsWith('@@'))cls='hunk';else if(s.startsWith('+')&&!s.startsWith('+++')){cls='add';added++;}else if(s.startsWith('-')&&!s.startsWith('---')){cls='del';removed++;}}return '<div id="L'+(i+1)+'" class="code-line '+cls+'"><a class="ln" href="#L'+(i+1)+'">'+(i+1)+'</a><span class="code-text">'+esc(s)+'</span></div>';}).join('');
  return (patch?'<div class="diff-summary">Diff · '+added+' addition'+(added===1?'':'s')+' · '+removed+' deletion'+(removed===1?'':'s')+'. Green: additions. Red: deletions.</div>':'')+'<div class="source">'+html+'</div>';
}
function breadcrumbs(name){const parts=name.split('/').filter(Boolean);let p='';$('breadcrumbs').innerHTML='<a href="/">comparisons</a>'+parts.map(s=>{p+=(p?'/':'')+s;return ' / <a href="'+route(p)+'">'+esc(s)+'</a>';}).join('');}
function renderFile(f, asSource=false){activeFile=f;const isMD=f.suffix==='.md',patch=f.suffix==='.patch';
  $('actions').innerHTML='<span>'+f.size.toLocaleString()+' bytes</span>'+((isMD&&f.text!==null)?'<button id="toggle">'+(asSource?'Preview':'Source')+'</button>':'')+'<a class="button" href="'+raw(f.path)+'">Download</a><button id="copy">Copy path</button>';
  $('copy').onclick=()=>navigator.clipboard.writeText(f.path).then(()=>{$('copy').textContent='Copied';});
  if($('toggle'))$('toggle').onclick=()=>renderFile(f,!asSource);
  if(f.text!==null)$('content').innerHTML=isMD&&!asSource?markdown(f.text,f.path):source(f.text,patch);
  else if(/\.(png|jpe?g)$/i.test(f.path))$('content').innerHTML='<img class="image-preview" src="'+raw(f.path)+'" alt="'+esc(f.path)+'">';
  else $('content').innerHTML='<div class="notice">Binary file or file larger than 2 MB. Download the original to inspect it. Sources and checksums are recorded in manifests/.</div>';
  highlightLine();
}
function highlightLine(){document.querySelectorAll('.code-line.selected').forEach(e=>e.classList.remove('selected'));if(location.hash){const element=document.getElementById(location.hash.slice(1));if(element){element.classList.add('selected');element.scrollIntoView({block:'center'});}}}
window.addEventListener('hashchange',highlightLine);
async function directory(name){const list=await api('tree',{path:name});$('actions').innerHTML='';$('content').innerHTML='<div class="tree">'+(name?'<a class="tree-row" href="'+route(name.split('/').slice(0,-1).join('/'))+'">↰ Parent directory</a>':'')+list.map(x=>'<a class="tree-row" href="'+route(x.path)+'"><span class="icon '+(x.directory?'directory':'')+'">'+(x.directory?'▣':'▤')+'</span>'+esc(x.name)+'</a>').join('')+'</div>';
  const readme=list.find(x=>x.name==='README.md')||list.find(x=>x.name==='README_CN.md');if(readme){const f=await api('file',{path:readme.path});if(f.text)$('content').insertAdjacentHTML('beforeend',markdown(f.text,readme.path));}}
async function load(){breadcrumbs(path);try{if(!path)await directory('');else{try{renderFile(await api('file',{path}),!!location.hash);}catch(e){await directory(path);}}}catch(e){$('content').innerHTML='<div class="notice">'+esc(e.message)+'</div>';}}
for(const name of ['cuthbert','pfjax','pydpf','tfp','particles','pypomp','pypfilt']){const div=document.createElement('div');div.className='pkg';div.innerHTML='<a href="'+route('packages/'+name)+'">'+esc(name)+'</a>'+(!['pypomp','pypfilt'].includes(name)?'<a href="'+route('packages/'+name+'/review/full.patch')+'">Diff</a>':'');$('packages').appendChild(div);}
api('meta').then(m=>{$('revision').textContent=m.branch+' · '+m.commit;$('count').textContent=m.files.toLocaleString()+' files';});
let timer;$('search').addEventListener('input',()=>{clearTimeout(timer);const q=$('search').value;const sequence=++requestSequence;timer=setTimeout(async()=>{if(!q){load();return;}const found=await api('search',{q});if(sequence!==requestSequence)return;$('breadcrumbs').textContent='File search · '+q;$('actions').textContent=found.total+' matches; showing up to 150';$('content').innerHTML='<div class="tree">'+found.paths.map(p=>'<a class="tree-row result-path" href="'+route(p)+'">'+esc(p)+'</a>').join('')+'</div>';},180);});
load();
