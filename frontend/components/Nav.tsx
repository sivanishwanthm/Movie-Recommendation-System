'use client';
import Link from 'next/link';
import {usePathname} from 'next/navigation';
export default function Nav(){const path=usePathname();return <header className="header"><Link href="/" className="brand" aria-label="RecSysLab home"><span className="brand-icon">r<span>·</span></span>RecSysLab<span className="brand-note">MOVIE DISCOVERY</span></Link><nav aria-label="Main navigation">{[['/','For you'],['/explore','Explore'],['/models','Models'],['/activity','Activity'],['/experiments','Experiments'],['/profile','Profile']].map(([url,title])=><Link key={url} href={url} aria-current={path===url?'page':undefined}>{title}</Link>)}</nav></header>}
