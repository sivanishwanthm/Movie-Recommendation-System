'use client';
import {useState} from 'react';
import {api,message,Profile} from '../lib/api';
export default function ProfileForm({onCreated}:{onCreated:(p:Profile)=>void}){
 const [name,setName]=useState('');const [error,setError]=useState('');const [busy,setBusy]=useState(false);
 return <section className="welcome panel"><div><span className="eyebrow">YOUR OWN STARTING POINT</span><h2>A little taste goes a long way.</h2><p>Create a profile, then rate a few movies you know. Your picks get more personal after three ratings of four stars or higher.</p><p className="small muted">Your private profile stays in this browser for 30 days. This local demo has no account recovery; clearing its cookie starts a new profile.</p></div><form onSubmit={async e=>{e.preventDefault();setBusy(true);setError('');try{onCreated(await api<Profile>('/profiles',{method:'POST',body:JSON.stringify({name})}));}catch(err){setError(message(err));}finally{setBusy(false);}}}><label htmlFor="profile-name">What should we call you?</label><input id="profile-name" required maxLength={60} value={name} placeholder="Your display name" onChange={e=>setName(e.target.value)}/><button disabled={busy||!name.trim()}>{busy?'Creating…':'Create my profile →'}</button>{error&&<p role="alert" className="error">{error}</p>}</form></section>
}
