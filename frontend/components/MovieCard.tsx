'use client';
import {useEffect,useRef,useState} from 'react';
import {useRouter} from 'next/navigation';
import Link from 'next/link';
import {Movie,api,impression,message} from '../lib/api';
export default function MovieCard({movie,index=0,batch}:{movie:Movie;index?:number;batch?:string}){
 const ref=useRef<HTMLElement>(null);const router=useRouter();const [error,setError]=useState('');const [busy,setBusy]=useState(false);
 useEffect(()=>{if(!batch||!ref.current)return;let logged=false;const observer=new IntersectionObserver(entries=>{if(entries.some(x=>x.isIntersecting)&&!logged){logged=true;impression(batch,movie.id).catch(()=>{logged=false;});}},{threshold:.5});observer.observe(ref.current);return()=>observer.disconnect();},[batch,movie.id]);
 const match=movie.title.match(/^(.*)\s\((\d{4})\)$/);const title=match?match[1]:movie.title;const year=match?match[2]:'';
 return <article ref={ref} className="movie-card"><Link className="movie-cover" href={`/movies/${movie.id}`} style={{'--cover-hue':String((movie.id*47)%360)} as React.CSSProperties} onClick={async e=>{if(!batch)return;e.preventDefault();if(busy)return;setBusy(true);setError('');try{await impression(batch,movie.id);await api('/clicks',{method:'POST',body:JSON.stringify({event_id:crypto.randomUUID(),batch_id:batch,movie_id:movie.id})});router.push(`/movies/${movie.id}`);}catch(err){setError(message(err));setBusy(false);}}} aria-label={`View ${movie.title}`}><span className="cover-top">RECSYSLAB COLLECTION <span>{String(index+1).padStart(2,'0')}</span></span><span className="cover-shape"/><span className="cover-title">{title}</span><span className="cover-bottom">{movie.genres[0]||'Film'} <span>{year}</span></span></Link><div className="movie-info"><h3>{title}</h3><p>{year}{year?' · ':''}{movie.genres.slice(0,2).join(' / ')}</p>{busy&&<span className="small">Opening…</span>}{error&&<p role="alert" className="error small">{error} Select the movie to retry.</p>}</div></article>
}
