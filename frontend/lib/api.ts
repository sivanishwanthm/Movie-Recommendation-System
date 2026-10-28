export type Movie = {id:number; title:string; genres:string[]; recommendable:boolean};
export type Profile = {id:string; name:string};
export type Rating = {movie:Movie; value:number; updated_at:string};
export type Recommendation = {batch_id:string; items:Movie[]; strategy:string; positive_count:number; model_version:string; generation_ms:number};
export class ApiError extends Error { constructor(public status:number,message:string){super(message);} }
export async function api<T>(path:string, options:RequestInit = {}):Promise<T> {
  const response = await fetch(`/api${path}`, {...options, credentials:'same-origin', cache:'no-store', headers:{'Content-Type':'application/json','X-Recsys-Client':'web',...options.headers}});
  const data = await response.json().catch(()=>null);
  if(!response.ok){
    const detail=typeof data?.detail==='string'?data.detail:response.status===422?'Please check the supplied values.':'The service is unavailable. Please retry.';
    throw new ApiError(response.status,detail);
  }
  return data as T;
}
export function message(error:unknown){return error instanceof Error?error.message:'Something went wrong. Please retry.';}
export async function impression(batch:string,id:number){return api('/impressions',{method:'POST',body:JSON.stringify({batch_id:batch,movie_ids:[id]})});}
