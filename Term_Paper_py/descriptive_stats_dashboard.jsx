import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell,
  Legend, ReferenceLine
} from "recharts";

const DATA = {"brand_counts":[{"brand":"thesouledstore","posts":194,"tier":"Core"},{"brand":"insightcosmetic","posts":185,"tier":"Core"},{"brand":"mcaffeineofficial","posts":171,"tier":"Core"},{"brand":"plumgoodness","posts":166,"tier":"Core"},{"brand":"bira91beer","posts":158,"tier":"Core"},{"brand":"bombayshirts","posts":156,"tier":"Core"},{"brand":"bombayshavingcompany","posts":154,"tier":"Core"},{"brand":"jimmysbeverages","posts":153,"tier":"Core"},{"brand":"foxtaleskin","posts":149,"tier":"Core"},{"brand":"sironahygiene","posts":131,"tier":"Core"},{"brand":"indya","posts":124,"tier":"Core"},{"brand":"myblissclub","posts":107,"tier":"Core"},{"brand":"beminimalist__","posts":83,"tier":"Strong"},{"brand":"thewholetruthfoods","posts":82,"tier":"Strong"},{"brand":"mokobara","posts":66,"tier":"Strong"},{"brand":"sleepyowlcoffee","posts":31,"tier":"Okay"},{"brand":"wakefitco","posts":11,"tier":"Weak"},{"brand":"suta_bombay","posts":7,"tier":"Weak"}],"brand_engagement":[{"brand":"beminimalist__","medLikes":932,"medComments":92,"medVideoViews":30345},{"brand":"bira91beer","medLikes":293,"medComments":3,"medVideoViews":2942},{"brand":"bombayshavingcompany","medLikes":266,"medComments":4,"medVideoViews":35616},{"brand":"bombayshirts","medLikes":743,"medComments":1,"medVideoViews":19060},{"brand":"foxtaleskin","medLikes":608,"medComments":57,"medVideoViews":13838},{"brand":"indya","medLikes":58,"medComments":2,"medVideoViews":16624},{"brand":"insightcosmetic","medLikes":742,"medComments":28,"medVideoViews":24262},{"brand":"jimmysbeverages","medLikes":51,"medComments":0,"medVideoViews":545},{"brand":"mcaffeineofficial","medLikes":393,"medComments":26,"medVideoViews":12806},{"brand":"mokobara","medLikes":731,"medComments":11,"medVideoViews":14146},{"brand":"myblissclub","medLikes":77,"medComments":10,"medVideoViews":6434},{"brand":"plumgoodness","medLikes":380,"medComments":38,"medVideoViews":10386},{"brand":"sironahygiene","medLikes":171,"medComments":7,"medVideoViews":5345},{"brand":"sleepyowlcoffee","medLikes":197,"medComments":7,"medVideoViews":3845},{"brand":"suta_bombay","medLikes":96,"medComments":11,"medVideoViews":3921},{"brand":"thesouledstore","medLikes":1682,"medComments":42,"medVideoViews":23652},{"brand":"thewholetruthfoods","medLikes":800,"medComments":42,"medVideoViews":11132},{"brand":"wakefitco","medLikes":40,"medComments":9,"medVideoViews":2114}],"type_dist":{"Video":1157,"Sidecar":600,"Image":371},"monthly":[{"month":"2024-09","posts":15},{"month":"2024-10","posts":24},{"month":"2024-11","posts":20},{"month":"2024-12","posts":6},{"month":"2025-01","posts":20},{"month":"2025-02","posts":20},{"month":"2025-03","posts":17},{"month":"2025-04","posts":19},{"month":"2025-05","posts":34},{"month":"2025-06","posts":28},{"month":"2025-07","posts":67},{"month":"2025-08","posts":125},{"month":"2025-09","posts":245},{"month":"2025-10","posts":218},{"month":"2025-11","posts":250},{"month":"2025-12","posts":282},{"month":"2026-01","posts":355},{"month":"2026-02","posts":224}],"likes_dist":[{"bucket":"0-100","count":402},{"bucket":"100-500","count":785},{"bucket":"500-1K","count":321},{"bucket":"1K-5K","count":465},{"bucket":"5K-10K","count":65},{"bucket":"10K-50K","count":61},{"bucket":"50K+","count":29}],"engagement_rate":[{"brand":"wakefitco","rate":0.176},{"brand":"suta_bombay","rate":0.111},{"brand":"myblissclub","rate":0.108},{"brand":"foxtaleskin","rate":0.096},{"brand":"beminimalist__","rate":0.084},{"brand":"plumgoodness","rate":0.082},{"brand":"mcaffeineofficial","rate":0.067},{"brand":"thewholetruthfoods","rate":0.056},{"brand":"sleepyowlcoffee","rate":0.042},{"brand":"sironahygiene","rate":0.040},{"brand":"insightcosmetic","rate":0.037},{"brand":"indya","rate":0.029},{"brand":"thesouledstore","rate":0.025},{"brand":"bombayshavingcompany","rate":0.013},{"brand":"bira91beer","rate":0.009}],"hourly":[{"hour":2,"posts":1},{"hour":3,"posts":6},{"hour":4,"posts":24},{"hour":5,"posts":63},{"hour":6,"posts":241},{"hour":7,"posts":126},{"hour":8,"posts":81},{"hour":9,"posts":76},{"hour":10,"posts":92},{"hour":11,"posts":134},{"hour":12,"posts":425},{"hour":13,"posts":414},{"hour":14,"posts":179},{"hour":15,"posts":135},{"hour":16,"posts":79},{"hour":17,"posts":30},{"hour":18,"posts":17},{"hour":19,"posts":4},{"hour":20,"posts":1}],"hashtag_dist":[{"count":0,"posts":1082},{"count":1,"posts":97},{"count":2,"posts":230},{"count":3,"posts":270},{"count":4,"posts":103},{"count":5,"posts":116},{"count":6,"posts":47},{"count":7,"posts":22},{"count":8,"posts":29},{"count":9,"posts":22},{"count":10,"posts":110}]};

const COLORS = {
  primary: "#FF6B35",
  secondary: "#004E89",
  accent: "#F7B801",
  dark: "#0A0A0A",
  light: "#F5F3EF",
  muted: "#6B717E"
};

const shortName = (s) => s.replace("official","").replace("hygiene","").replace("beverages","").replace("goodness","").replace("cosmetic","").replace("company","");

export default function DescriptiveStats() {
  const [view, setView] = useState("overview");

  const topBrands = DATA.brand_engagement.sort((a,b) => b.medLikes - a.medLikes).slice(0,12);
  const typePie = [
    {name:"Video",value:DATA.type_dist.Video,fill:COLORS.primary},
    {name:"Sidecar",value:DATA.type_dist.Sidecar,fill:COLORS.secondary},
    {name:"Image",value:DATA.type_dist.Image,fill:COLORS.accent}
  ];

  return (
    <div style={{fontFamily:"'Playfair Display', Georgia, serif",background:`linear-gradient(135deg, ${COLORS.dark} 0%, #1a1a1a 100%)`,minHeight:"100vh",color:COLORS.light,padding:0}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Work+Sans:wght@300;400;500;600&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        body { overflow-x:hidden; }
        .headline { font-size:clamp(36px,5vw,64px); font-weight:900; line-height:1.1; letter-spacing:-2px; margin-bottom:8px; background:linear-gradient(120deg,#FF6B35,#F7B801); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
        .subhead { font-family:'Work Sans',sans-serif; font-size:clamp(14px,1.5vw,18px); color:#999; letter-spacing:3px; text-transform:uppercase; font-weight:300; }
        .stat-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:24px; backdrop-filter:blur(10px); transition:all 0.3s; }
        .stat-card:hover { background:rgba(255,255,255,0.05); border-color:rgba(255,255,255,0.15); transform:translateY(-2px); }
        .stat-big { font-family:'Work Sans',sans-serif; font-size:48px; font-weight:700; color:#FF6B35; line-height:1; margin-bottom:8px; }
        .stat-label { font-family:'Work Sans',sans-serif; font-size:12px; color:#666; letter-spacing:2px; text-transform:uppercase; }
        .section-title { font-size:28px; font-weight:700; letter-spacing:-1px; margin-bottom:24px; color:#F5F3EF; }
        .badge { display:inline-block; padding:4px 12px; border-radius:20px; font-family:'Work Sans',sans-serif; font-size:11px; font-weight:600; letter-spacing:1px; }
        .nav-btn { padding:10px 20px; border:none; background:transparent; color:#888; font-family:'Work Sans',sans-serif; font-size:13px; letter-spacing:2px; text-transform:uppercase; cursor:pointer; transition:all 0.2s; border-bottom:2px solid transparent; }
        .nav-btn:hover { color:#F5F3EF; }
        .nav-btn.active { color:#FF6B35; border-bottom-color:#FF6B35; }
      `}</style>

      {/* Hero Header */}
      <div style={{borderBottom:"1px solid rgba(255,255,255,0.08)",padding:"48px 40px 36px"}}>
        <div className="subhead" style={{marginBottom:12}}>Indian D2C · Instagram Dataset</div>
        <div className="headline">Descriptive Statistics</div>
        <div style={{fontFamily:"'Work Sans',sans-serif",fontSize:15,color:"#777",marginTop:16,maxWidth:600}}>
          Deep dive into 2,128 Instagram posts from 18 Indian D2C brands spanning August 2023–February 2026
        </div>
        
        {/* Nav */}
        <div style={{display:"flex",gap:8,marginTop:32,borderTop:"1px solid rgba(255,255,255,0.05)",paddingTop:16}}>
          {["overview","engagement","content","timing"].map(v => (
            <button key={v} className={`nav-btn ${view===v?"active":""}`} onClick={()=>setView(v)}>{v}</button>
          ))}
        </div>
      </div>

      <div style={{padding:"40px"}}>

        {/* OVERVIEW */}
        {view==="overview" && (
          <div>
            {/* Top KPIs */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:16,marginBottom:40}}>
              <div className="stat-card">
                <div className="stat-big">2,128</div>
                <div className="stat-label">Total Posts</div>
              </div>
              <div className="stat-card">
                <div className="stat-big">18</div>
                <div className="stat-label">Brands</div>
              </div>
              <div className="stat-card">
                <div className="stat-big">398</div>
                <div className="stat-label">Median Likes</div>
              </div>
              <div className="stat-card">
                <div className="stat-big">16</div>
                <div className="stat-label">Median Comments</div>
              </div>
              <div className="stat-card">
                <div className="stat-big">927</div>
                <div className="stat-label">Days Covered</div>
              </div>
            </div>

            {/* Brand Coverage + Type Distribution */}
            <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:20,marginBottom:40}}>
              <div className="stat-card">
                <div className="section-title">Brand Post Volume</div>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={DATA.brand_counts} layout="vertical" margin={{left:120}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                    <YAxis type="category" dataKey="brand" tick={{fill:"#999",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} tickFormatter={shortName} width={110} />
                    <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                    <ReferenceLine x={100} stroke="#FF6B35" strokeDasharray="4 4" label={{value:"100",fill:"#FF6B35",fontSize:11,position:"top"}} />
                    <Bar dataKey="posts" radius={[0,4,4,0]}>
                      {DATA.brand_counts.map(d => (
                        <Cell key={d.brand} fill={d.tier==="Core"?"#FF6B35":d.tier==="Strong"?"#F7B801":"#555"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{display:"flex",gap:16,marginTop:16,fontFamily:"'Work Sans',sans-serif",fontSize:11}}>
                  <div><span style={{color:"#FF6B35"}}>█</span> Core (100+)</div>
                  <div><span style={{color:"#F7B801"}}>█</span> Strong (50–99)</div>
                  <div><span style={{color:"#555"}}>█</span> Weak (<50)</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="section-title">Post Type Mix</div>
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={typePie} cx="50%" cy="50%" outerRadius={80} innerRadius={50} dataKey="value" paddingAngle={4}>
                      {typePie.map((d,i) => <Cell key={i} fill={d.fill} />)}
                    </Pie>
                    <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{display:"flex",flexDirection:"column",gap:8,marginTop:16}}>
                  {typePie.map(d => (
                    <div key={d.name} style={{display:"flex",justifyContent:"space-between",fontFamily:"'Work Sans',sans-serif",fontSize:13}}>
                      <span style={{display:"flex",alignItems:"center",gap:8}}>
                        <span style={{width:10,height:10,borderRadius:"50%",background:d.fill,display:"inline-block"}} />
                        {d.name}
                      </span>
                      <span style={{color:"#666"}}>{d.value} ({(d.value/21.28).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Monthly Trend */}
            <div className="stat-card" style={{marginBottom:40}}>
              <div className="section-title">Monthly Posting Trend</div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={DATA.monthly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="month" tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <YAxis tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                  <Line type="monotone" dataKey="posts" stroke="#FF6B35" strokeWidth={3} dot={{r:4,fill:"#FF6B35",strokeWidth:2,stroke:"#1a1a1a"}} />
                </LineChart>
              </ResponsiveContainer>
              <div style={{marginTop:12,padding:12,background:"rgba(255,107,53,0.1)",borderLeft:"3px solid #FF6B35",borderRadius:4,fontFamily:"'Work Sans',sans-serif",fontSize:12,color:"#999"}}>
                📈 Sharp increase from Aug 2025 onwards — scraper successfully captured recent activity
              </div>
            </div>

            {/* Likes Distribution */}
            <div className="stat-card">
              <div className="section-title">Likes Distribution (Log Scale)</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={DATA.likes_dist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="bucket" tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <YAxis tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                  <Bar dataKey="count" radius={[4,4,0,0]}>
                    {DATA.likes_dist.map((_,i) => <Cell key={i} fill={i===1?"#FF6B35":"#333"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{marginTop:12,padding:12,background:"rgba(247,184,1,0.1)",borderLeft:"3px solid #F7B801",borderRadius:4,fontFamily:"'Work Sans',sans-serif",fontSize:12,color:"#999"}}>
                ⚠️ Right-skewed distribution — log transformation required before regression analysis
              </div>
            </div>
          </div>
        )}

        {/* ENGAGEMENT */}
        {view==="engagement" && (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:40}}>
              <div className="stat-card">
                <div className="section-title">Top 12 Brands by Median Likes</div>
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={topBrands} layout="vertical" margin={{left:120}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" tick={{fill:"#666",fontSize:10,fontFamily:"'Work Sans',sans-serif"}} />
                    <YAxis type="category" dataKey="brand" tick={{fill:"#999",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} tickFormatter={shortName} width={110} />
                    <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                    <Bar dataKey="medLikes" radius={[0,4,4,0]} fill="#004E89" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="stat-card">
                <div className="section-title">Comment/Like Ratio (Top 15)</div>
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={DATA.engagement_rate} layout="vertical" margin={{left:120}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" tickFormatter={v=>`${(v*100).toFixed(1)}%`} tick={{fill:"#666",fontSize:10,fontFamily:"'Work Sans',sans-serif"}} />
                    <YAxis type="category" dataKey="brand" tick={{fill:"#999",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} tickFormatter={shortName} width={110} />
                    <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} formatter={v=>`${(v*100).toFixed(2)}%`} />
                    <Bar dataKey="rate" radius={[0,4,4,0]}>
                      {DATA.engagement_rate.map((d,i) => (
                        <Cell key={i} fill={d.rate>0.08?"#FF6B35":d.rate>0.04?"#F7B801":"#555"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Engagement Table */}
            <div className="stat-card">
              <div className="section-title">Brand Engagement Summary</div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"'Work Sans',sans-serif",fontSize:13}}>
                  <thead>
                    <tr style={{borderBottom:"2px solid rgba(255,255,255,0.1)"}}>
                      <th style={{textAlign:"left",padding:"12px",color:"#666",fontWeight:600,letterSpacing:"1px",fontSize:11,textTransform:"uppercase"}}>Brand</th>
                      <th style={{textAlign:"right",padding:"12px",color:"#666",fontWeight:600,letterSpacing:"1px",fontSize:11,textTransform:"uppercase"}}>Med. Likes</th>
                      <th style={{textAlign:"right",padding:"12px",color:"#666",fontWeight:600,letterSpacing:"1px",fontSize:11,textTransform:"uppercase"}}>Med. Comments</th>
                      <th style={{textAlign:"right",padding:"12px",color:"#666",fontWeight:600,letterSpacing:"1px",fontSize:11,textTransform:"uppercase"}}>Med. Video Views</th>
                      <th style={{textAlign:"right",padding:"12px",color:"#666",fontWeight:600,letterSpacing:"1px",fontSize:11,textTransform:"uppercase"}}>Eng. Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {DATA.brand_engagement.sort((a,b)=>b.medLikes-a.medLikes).map(b => {
                      const engRate = DATA.engagement_rate.find(e=>e.brand===b.brand)?.rate || 0;
                      return (
                        <tr key={b.brand} style={{borderBottom:"1px solid rgba(255,255,255,0.05)"}}>
                          <td style={{padding:"10px",color:"#F5F3EF"}}>{b.brand}</td>
                          <td style={{padding:"10px",textAlign:"right",color:"#FF6B35",fontWeight:600}}>{b.medLikes.toLocaleString()}</td>
                          <td style={{padding:"10px",textAlign:"right",color:"#999"}}>{b.medComments}</td>
                          <td style={{padding:"10px",textAlign:"right",color:"#999"}}>{b.medVideoViews?.toLocaleString() || "—"}</td>
                          <td style={{padding:"10px",textAlign:"right",color:engRate>0.08?"#FF6B35":"#666"}}>{(engRate*100).toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* CONTENT */}
        {view==="content" && (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:40}}>
              <div className="stat-card">
                <div className="section-title">Caption Length Stats</div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginTop:20}}>
                  <div>
                    <div style={{fontSize:40,fontWeight:700,color:"#FF6B35",fontFamily:"'Work Sans',sans-serif"}}>221</div>
                    <div className="stat-label">Median Chars</div>
                  </div>
                  <div>
                    <div style={{fontSize:40,fontWeight:700,color:"#004E89",fontFamily:"'Work Sans',sans-serif"}}>285</div>
                    <div className="stat-label">Mean Chars</div>
                  </div>
                  <div>
                    <div style={{fontSize:40,fontWeight:700,color:"#F7B801",fontFamily:"'Work Sans',sans-serif"}}>2</div>
                    <div className="stat-label">Empty Captions</div>
                  </div>
                  <div>
                    <div style={{fontSize:40,fontWeight:700,color:"#666",fontFamily:"'Work Sans',sans-serif"}}>0.09%</div>
                    <div className="stat-label">Empty Rate</div>
                  </div>
                </div>
                <div style={{marginTop:24,padding:12,background:"rgba(0,78,137,0.1)",borderLeft:"3px solid #004E89",borderRadius:4,fontFamily:"'Work Sans',sans-serif",fontSize:12,color:"#999"}}>
                  ✓ 99.9% caption completeness — excellent for text analysis
                </div>
              </div>

              <div className="stat-card">
                <div className="section-title">Hashtag Usage Distribution</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={DATA.hashtag_dist}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="count" label={{value:"# Hashtags",position:"insideBottom",fill:"#666",dy:10,fontFamily:"'Work Sans',sans-serif",fontSize:11}} tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                    <YAxis tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                    <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                    <Bar dataKey="posts" radius={[4,4,0,0]}>
                      {DATA.hashtag_dist.map((d,i) => <Cell key={i} fill={d.count===0?"#555":"#F7B801"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{marginTop:12,padding:12,background:"rgba(247,184,1,0.1)",borderLeft:"3px solid #F7B801",borderRadius:4,fontFamily:"'Work Sans',sans-serif",fontSize:12,color:"#999"}}>
                  ⚠️ 51% of posts have 0 hashtags — may limit hashtag analysis
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TIMING */}
        {view==="timing" && (
          <div>
            <div className="stat-card" style={{marginBottom:40}}>
              <div className="section-title">Hourly Posting Pattern (UTC)</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={DATA.hourly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="hour" label={{value:"Hour of Day (UTC)",position:"insideBottom",fill:"#666",dy:10,fontFamily:"'Work Sans',sans-serif",fontSize:11}} tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <YAxis tick={{fill:"#666",fontSize:11,fontFamily:"'Work Sans',sans-serif"}} />
                  <Tooltip contentStyle={{background:"#1a1a1a",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,fontFamily:"'Work Sans',sans-serif",fontSize:12}} />
                  <Bar dataKey="posts" radius={[4,4,0,0]}>
                    {DATA.hourly.map((d,i) => (
                      <Cell key={i} fill={d.hour>=12 && d.hour<=13?"#FF6B35":"#555"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{marginTop:16,padding:12,background:"rgba(255,107,53,0.1)",borderLeft:"3px solid #FF6B35",borderRadius:4,fontFamily:"'Work Sans',sans-serif",fontSize:12,color:"#999"}}>
                📌 Peak posting: 12–1 PM UTC (5:30–6:30 PM IST) — prime evening engagement window
              </div>
            </div>

            <div className="stat-card">
              <div className="section-title">Key Temporal Insights</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16,marginTop:16}}>
                <div style={{padding:20,background:"rgba(255,107,53,0.05)",border:"1px solid rgba(255,107,53,0.2)",borderRadius:8}}>
                  <div style={{fontSize:14,color:"#666",marginBottom:8,fontFamily:"'Work Sans',sans-serif",textTransform:"uppercase",letterSpacing:"1px"}}>Most Active Month</div>
                  <div style={{fontSize:32,fontWeight:700,color:"#FF6B35",fontFamily:"'Work Sans',sans-serif"}}>Jan 2026</div>
                  <div style={{fontSize:13,color:"#999",marginTop:4,fontFamily:"'Work Sans',sans-serif"}}>355 posts</div>
                </div>
                <div style={{padding:20,background:"rgba(0,78,137,0.05)",border:"1px solid rgba(0,78,137,0.2)",borderRadius:8}}>
                  <div style={{fontSize:14,color:"#666",marginBottom:8,fontFamily:"'Work Sans',sans-serif",textTransform:"uppercase",letterSpacing:"1px"}}>Peak Hour</div>
                  <div style={{fontSize:32,fontWeight:700,color:"#004E89",fontFamily:"'Work Sans',sans-serif"}}>12 PM</div>
                  <div style={{fontSize:13,color:"#999",marginTop:4,fontFamily:"'Work Sans',sans-serif"}}>425 posts (20%)</div>
                </div>
                <div style={{padding:20,background:"rgba(247,184,1,0.05)",border:"1px solid rgba(247,184,1,0.2)",borderRadius:8}}>
                  <div style={{fontSize:14,color:"#666",marginBottom:8,fontFamily:"'Work Sans',sans-serif",textTransform:"uppercase",letterSpacing:"1px"}}>Least Active</div>
                  <div style={{fontSize:32,fontWeight:700,color:"#F7B801",fontFamily:"'Work Sans',sans-serif"}}>Dec 2024</div>
                  <div style={{fontSize:13,color:"#999",marginTop:4,fontFamily:"'Work Sans',sans-serif"}}>6 posts only</div>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
