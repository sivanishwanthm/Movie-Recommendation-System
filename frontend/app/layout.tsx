import type {Metadata} from 'next';
import Nav from '../components/Nav';
import './globals.css';
export const metadata:Metadata={title:'RecSysLab · Find your next favorite',description:'Movie discovery powered by real recommendation models and your ratings.'};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body><a className="skip-link" href="#main">Skip to content</a><Nav/><main id="main">{children}</main><footer><span>RecSysLab <b> / </b> A recommendation research project</span><span>Movie metadata: GroupLens MovieLens · No streaming available</span></footer></body></html>}
