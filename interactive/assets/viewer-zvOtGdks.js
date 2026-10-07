/**
 * @license
 * Copyright 2010-2023 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const Sn={ROTATE:0,DOLLY:1,PAN:2},yn={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},no=0,Zr=1,io=2,xa=1,Ma=2,ei=3,en=0,pe=1,Xe=2,tn=0,Vn=1,Kr=2,$r=3,Qr=4,ro=5,Bn=100,so=101,ao=102,ts=103,es=104,oo=200,lo=201,co=202,ho=203,Sa=204,ya=205,uo=206,fo=207,po=208,mo=209,go=210,_o=0,vo=1,xo=2,Tr=3,Mo=4,So=5,yo=6,bo=7,ba=0,wo=1,Eo=2,je=0,To=1,Ao=2,Co=3,wa=4,Lo=5,Ea=300,Wn=301,qn=302,Ar=303,Cr=304,Oi=306,Lr=1e3,Re=1001,Pr=1002,fe=1003,ns=1004,Wi=1005,be=1006,Po=1007,si=1008,mn=1009,Ro=1010,Do=1011,Ta=1012,Io=1013,dn=1014,fn=1015,ai=1016,Uo=1017,No=1018,kn=1020,Oo=1021,De=1023,Fo=1024,zo=1025,pn=1026,Xn=1027,Bo=1028,Go=1029,Vo=1030,ko=1031,Ho=1033,qi=33776,Xi=33777,Yi=33778,ji=33779,is=35840,rs=35841,ss=35842,as=35843,Wo=36196,os=37492,ls=37496,cs=37808,hs=37809,us=37810,ds=37811,fs=37812,ps=37813,ms=37814,gs=37815,_s=37816,vs=37817,xs=37818,Ms=37819,Ss=37820,ys=37821,Ji=36492,qo=36283,bs=36284,ws=36285,Es=36286,gn=3e3,Gt=3001,Xo=3200,Yo=3201,Aa=0,jo=1,Oe="srgb",oi="srgb-linear",Ca="display-p3",Zi=7680,Jo=519,Ts=35044,As="300 es",Rr=1035;class Mn{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[t]===void 0&&(n[t]=[]),n[t].indexOf(e)===-1&&n[t].push(e)}hasEventListener(t,e){if(this._listeners===void 0)return!1;const n=this._listeners;return n[t]!==void 0&&n[t].indexOf(e)!==-1}removeEventListener(t,e){if(this._listeners===void 0)return;const i=this._listeners[t];if(i!==void 0){const r=i.indexOf(e);r!==-1&&i.splice(r,1)}}dispatchEvent(t){if(this._listeners===void 0)return;const n=this._listeners[t.type];if(n!==void 0){t.target=this;const i=n.slice(0);for(let r=0,o=i.length;r<o;r++)i[r].call(this,t);t.target=null}}}const le=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],Ki=Math.PI/180,Dr=180/Math.PI;function li(){const a=Math.random()*4294967295|0,t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(le[a&255]+le[a>>8&255]+le[a>>16&255]+le[a>>24&255]+"-"+le[t&255]+le[t>>8&255]+"-"+le[t>>16&15|64]+le[t>>24&255]+"-"+le[e&63|128]+le[e>>8&255]+"-"+le[e>>16&255]+le[e>>24&255]+le[n&255]+le[n>>8&255]+le[n>>16&255]+le[n>>24&255]).toLowerCase()}function ne(a,t,e){return Math.max(t,Math.min(e,a))}function Zo(a,t){return(a%t+t)%t}function $i(a,t,e){return(1-e)*a+e*t}function Cs(a){return(a&a-1)===0&&a!==0}function Ko(a){return Math.pow(2,Math.floor(Math.log(a)/Math.LN2))}function di(a,t){switch(t.constructor){case Float32Array:return a;case Uint16Array:return a/65535;case Uint8Array:return a/255;case Int16Array:return Math.max(a/32767,-1);case Int8Array:return Math.max(a/127,-1);default:throw new Error("Invalid component type.")}}function ge(a,t){switch(t.constructor){case Float32Array:return a;case Uint16Array:return Math.round(a*65535);case Uint8Array:return Math.round(a*255);case Int16Array:return Math.round(a*32767);case Int8Array:return Math.round(a*127);default:throw new Error("Invalid component type.")}}class dt{constructor(t=0,e=0){dt.prototype.isVector2=!0,this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t){return this.x+=t.x,this.y+=t.y,this}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){const e=this.x,n=this.y,i=t.elements;return this.x=i[0]*e+i[3]*n+i[6],this.y=i[1]*e+i[4]*n+i[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(ne(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y;return e*e+n*n}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){const n=Math.cos(e),i=Math.sin(e),r=this.x-t.x,o=this.y-t.y;return this.x=r*n-o*i+t.x,this.y=r*i+o*n+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class It{constructor(){It.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1]}set(t,e,n,i,r,o,s,c,l){const h=this.elements;return h[0]=t,h[1]=i,h[2]=s,h[3]=e,h[4]=r,h[5]=c,h[6]=n,h[7]=o,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],this}extractBasis(t,e,n){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(t){const e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,i=e.elements,r=this.elements,o=n[0],s=n[3],c=n[6],l=n[1],h=n[4],d=n[7],f=n[2],p=n[5],g=n[8],_=i[0],m=i[3],u=i[6],E=i[1],x=i[4],b=i[7],y=i[2],C=i[5],D=i[8];return r[0]=o*_+s*E+c*y,r[3]=o*m+s*x+c*C,r[6]=o*u+s*b+c*D,r[1]=l*_+h*E+d*y,r[4]=l*m+h*x+d*C,r[7]=l*u+h*b+d*D,r[2]=f*_+p*E+g*y,r[5]=f*m+p*x+g*C,r[8]=f*u+p*b+g*D,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],o=t[4],s=t[5],c=t[6],l=t[7],h=t[8];return e*o*h-e*s*l-n*r*h+n*s*c+i*r*l-i*o*c}invert(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],o=t[4],s=t[5],c=t[6],l=t[7],h=t[8],d=h*o-s*l,f=s*c-h*r,p=l*r-o*c,g=e*d+n*f+i*p;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);const _=1/g;return t[0]=d*_,t[1]=(i*l-h*n)*_,t[2]=(s*n-i*o)*_,t[3]=f*_,t[4]=(h*e-i*c)*_,t[5]=(i*r-s*e)*_,t[6]=p*_,t[7]=(n*c-l*e)*_,t[8]=(o*e-n*r)*_,this}transpose(){let t;const e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){const e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,n,i,r,o,s){const c=Math.cos(r),l=Math.sin(r);return this.set(n*c,n*l,-n*(c*o+l*s)+o+t,-i*l,i*c,-i*(-l*o+c*s)+s+e,0,0,1),this}scale(t,e){return this.premultiply(Qi.makeScale(t,e)),this}rotate(t){return this.premultiply(Qi.makeRotation(-t)),this}translate(t,e){return this.premultiply(Qi.makeTranslation(t,e)),this}makeTranslation(t,e){return this.set(1,0,t,0,1,e,0,0,1),this}makeRotation(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,n,e,0,0,0,1),this}makeScale(t,e){return this.set(t,0,0,0,e,0,0,0,1),this}equals(t){const e=this.elements,n=t.elements;for(let i=0;i<9;i++)if(e[i]!==n[i])return!1;return!0}fromArray(t,e=0){for(let n=0;n<9;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t}clone(){return new this.constructor().fromArray(this.elements)}}const Qi=new It;function La(a){for(let t=a.length-1;t>=0;--t)if(a[t]>=65535)return!0;return!1}function Ni(a){return document.createElementNS("http://www.w3.org/1999/xhtml",a)}function Hn(a){return a<.04045?a*.0773993808:Math.pow(a*.9478672986+.0521327014,2.4)}function tr(a){return a<.0031308?a*12.92:1.055*Math.pow(a,.41666)-.055}const $o=new It().fromArray([.8224621,.0331941,.0170827,.177538,.9668058,.0723974,-1e-7,1e-7,.9105199]),Qo=new It().fromArray([1.2249401,-.0420569,-.0196376,-.2249404,1.0420571,-.0786361,1e-7,0,1.0982735]);function tl(a){return a.convertSRGBToLinear().applyMatrix3(Qo)}function el(a){return a.applyMatrix3($o).convertLinearToSRGB()}const nl={[oi]:a=>a,[Oe]:a=>a.convertSRGBToLinear(),[Ca]:tl},il={[oi]:a=>a,[Oe]:a=>a.convertLinearToSRGB(),[Ca]:el},_e={enabled:!1,get legacyMode(){return console.warn("THREE.ColorManagement: .legacyMode=false renamed to .enabled=true in r150."),!this.enabled},set legacyMode(a){console.warn("THREE.ColorManagement: .legacyMode=false renamed to .enabled=true in r150."),this.enabled=!a},get workingColorSpace(){return oi},set workingColorSpace(a){console.warn("THREE.ColorManagement: .workingColorSpace is readonly.")},convert:function(a,t,e){if(this.enabled===!1||t===e||!t||!e)return a;const n=nl[t],i=il[e];if(n===void 0||i===void 0)throw new Error(`Unsupported color space conversion, "${t}" to "${e}".`);return i(n(a))},fromWorkingColorSpace:function(a,t){return this.convert(a,this.workingColorSpace,t)},toWorkingColorSpace:function(a,t){return this.convert(a,t,this.workingColorSpace)}};let bn;class Pa{static getDataURL(t){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{bn===void 0&&(bn=Ni("canvas")),bn.width=t.width,bn.height=t.height;const n=bn.getContext("2d");t instanceof ImageData?n.putImageData(t,0,0):n.drawImage(t,0,0,t.width,t.height),e=bn}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){const e=Ni("canvas");e.width=t.width,e.height=t.height;const n=e.getContext("2d");n.drawImage(t,0,0,t.width,t.height);const i=n.getImageData(0,0,t.width,t.height),r=i.data;for(let o=0;o<r.length;o++)r[o]=Hn(r[o]/255)*255;return n.putImageData(i,0,0),e}else if(t.data){const e=t.data.slice(0);for(let n=0;n<e.length;n++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[n]=Math.floor(Hn(e[n]/255)*255):e[n]=Hn(e[n]);return{data:e,width:t.width,height:t.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}}class Ra{constructor(t=null){this.isSource=!0,this.uuid=li(),this.data=t,this.version=0}set needsUpdate(t){t===!0&&this.version++}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];const n={uuid:this.uuid,url:""},i=this.data;if(i!==null){let r;if(Array.isArray(i)){r=[];for(let o=0,s=i.length;o<s;o++)i[o].isDataTexture?r.push(er(i[o].image)):r.push(er(i[o]))}else r=er(i);n.url=r}return e||(t.images[this.uuid]=n),n}}function er(a){return typeof HTMLImageElement<"u"&&a instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&a instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&a instanceof ImageBitmap?Pa.getDataURL(a):a.data?{data:Array.from(a.data),width:a.width,height:a.height,type:a.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}let rl=0;class Me extends Mn{constructor(t=Me.DEFAULT_IMAGE,e=Me.DEFAULT_MAPPING,n=Re,i=Re,r=be,o=si,s=De,c=mn,l=Me.DEFAULT_ANISOTROPY,h=gn){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:rl++}),this.uuid=li(),this.name="",this.source=new Ra(t),this.mipmaps=[],this.mapping=e,this.channel=0,this.wrapS=n,this.wrapT=i,this.magFilter=r,this.minFilter=o,this.anisotropy=l,this.format=s,this.internalFormat=null,this.type=c,this.offset=new dt(0,0),this.repeat=new dt(1,1),this.center=new dt(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new It,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.encoding=h,this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1,this.needsPMREMUpdate=!1}get image(){return this.source.data}set image(t=null){this.source.data=t}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(t){return this.name=t.name,this.source=t.source,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.channel=t.channel,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.encoding=t.encoding,this.userData=JSON.parse(JSON.stringify(t.userData)),this.needsUpdate=!0,this}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.textures[this.uuid]!==void 0)return t.textures[this.uuid];const n={metadata:{version:4.5,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(t).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,encoding:this.encoding,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),e||(t.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==Ea)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case Lr:t.x=t.x-Math.floor(t.x);break;case Re:t.x=t.x<0?0:1;break;case Pr:Math.abs(Math.floor(t.x)%2)===1?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x);break}if(t.y<0||t.y>1)switch(this.wrapT){case Lr:t.y=t.y-Math.floor(t.y);break;case Re:t.y=t.y<0?0:1;break;case Pr:Math.abs(Math.floor(t.y)%2)===1?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y);break}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){t===!0&&(this.version++,this.source.needsUpdate=!0)}}Me.DEFAULT_IMAGE=null;Me.DEFAULT_MAPPING=Ea;Me.DEFAULT_ANISOTROPY=1;class ie{constructor(t=0,e=0,n=0,i=1){ie.prototype.isVector4=!0,this.x=t,this.y=e,this.z=n,this.w=i}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,n,i){return this.x=t,this.y=e,this.z=n,this.w=i,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=t.w!==void 0?t.w:1,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){const e=this.x,n=this.y,i=this.z,r=this.w,o=t.elements;return this.x=o[0]*e+o[4]*n+o[8]*i+o[12]*r,this.y=o[1]*e+o[5]*n+o[9]*i+o[13]*r,this.z=o[2]*e+o[6]*n+o[10]*i+o[14]*r,this.w=o[3]*e+o[7]*n+o[11]*i+o[15]*r,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);const e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,n,i,r;const c=t.elements,l=c[0],h=c[4],d=c[8],f=c[1],p=c[5],g=c[9],_=c[2],m=c[6],u=c[10];if(Math.abs(h-f)<.01&&Math.abs(d-_)<.01&&Math.abs(g-m)<.01){if(Math.abs(h+f)<.1&&Math.abs(d+_)<.1&&Math.abs(g+m)<.1&&Math.abs(l+p+u-3)<.1)return this.set(1,0,0,0),this;e=Math.PI;const x=(l+1)/2,b=(p+1)/2,y=(u+1)/2,C=(h+f)/4,D=(d+_)/4,R=(g+m)/4;return x>b&&x>y?x<.01?(n=0,i=.707106781,r=.707106781):(n=Math.sqrt(x),i=C/n,r=D/n):b>y?b<.01?(n=.707106781,i=0,r=.707106781):(i=Math.sqrt(b),n=C/i,r=R/i):y<.01?(n=.707106781,i=.707106781,r=0):(r=Math.sqrt(y),n=D/r,i=R/r),this.set(n,i,r,e),this}let E=Math.sqrt((m-g)*(m-g)+(d-_)*(d-_)+(f-h)*(f-h));return Math.abs(E)<.001&&(E=1),this.x=(m-g)/E,this.y=(d-_)/E,this.z=(f-h)/E,this.w=Math.acos((l+p+u-1)/2),this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this.w=Math.max(t.w,Math.min(e.w,this.w)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this.w=Math.max(t,Math.min(e,this.w)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this.z=this.z<0?Math.ceil(this.z):Math.floor(this.z),this.w=this.w<0?Math.ceil(this.w):Math.floor(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this.w=t.w+(e.w-t.w)*n,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class _n extends Mn{constructor(t=1,e=1,n={}){super(),this.isWebGLRenderTarget=!0,this.width=t,this.height=e,this.depth=1,this.scissor=new ie(0,0,t,e),this.scissorTest=!1,this.viewport=new ie(0,0,t,e);const i={width:t,height:e,depth:1};this.texture=new Me(i,n.mapping,n.wrapS,n.wrapT,n.magFilter,n.minFilter,n.format,n.type,n.anisotropy,n.encoding),this.texture.isRenderTargetTexture=!0,this.texture.flipY=!1,this.texture.generateMipmaps=n.generateMipmaps!==void 0?n.generateMipmaps:!1,this.texture.internalFormat=n.internalFormat!==void 0?n.internalFormat:null,this.texture.minFilter=n.minFilter!==void 0?n.minFilter:be,this.depthBuffer=n.depthBuffer!==void 0?n.depthBuffer:!0,this.stencilBuffer=n.stencilBuffer!==void 0?n.stencilBuffer:!1,this.depthTexture=n.depthTexture!==void 0?n.depthTexture:null,this.samples=n.samples!==void 0?n.samples:0}setSize(t,e,n=1){(this.width!==t||this.height!==e||this.depth!==n)&&(this.width=t,this.height=e,this.depth=n,this.texture.image.width=t,this.texture.image.height=e,this.texture.image.depth=n,this.dispose()),this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.viewport.copy(t.viewport),this.texture=t.texture.clone(),this.texture.isRenderTargetTexture=!0;const e=Object.assign({},t.texture.image);return this.texture.source=new Ra(e),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,t.depthTexture!==null&&(this.depthTexture=t.depthTexture.clone()),this.samples=t.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Da extends Me{constructor(t=null,e=1,n=1,i=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:n,depth:i},this.magFilter=fe,this.minFilter=fe,this.wrapR=Re,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class sl extends Me{constructor(t=null,e=1,n=1,i=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:n,depth:i},this.magFilter=fe,this.minFilter=fe,this.wrapR=Re,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class vn{constructor(t=0,e=0,n=0,i=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=n,this._w=i}static slerpFlat(t,e,n,i,r,o,s){let c=n[i+0],l=n[i+1],h=n[i+2],d=n[i+3];const f=r[o+0],p=r[o+1],g=r[o+2],_=r[o+3];if(s===0){t[e+0]=c,t[e+1]=l,t[e+2]=h,t[e+3]=d;return}if(s===1){t[e+0]=f,t[e+1]=p,t[e+2]=g,t[e+3]=_;return}if(d!==_||c!==f||l!==p||h!==g){let m=1-s;const u=c*f+l*p+h*g+d*_,E=u>=0?1:-1,x=1-u*u;if(x>Number.EPSILON){const y=Math.sqrt(x),C=Math.atan2(y,u*E);m=Math.sin(m*C)/y,s=Math.sin(s*C)/y}const b=s*E;if(c=c*m+f*b,l=l*m+p*b,h=h*m+g*b,d=d*m+_*b,m===1-s){const y=1/Math.sqrt(c*c+l*l+h*h+d*d);c*=y,l*=y,h*=y,d*=y}}t[e]=c,t[e+1]=l,t[e+2]=h,t[e+3]=d}static multiplyQuaternionsFlat(t,e,n,i,r,o){const s=n[i],c=n[i+1],l=n[i+2],h=n[i+3],d=r[o],f=r[o+1],p=r[o+2],g=r[o+3];return t[e]=s*g+h*d+c*p-l*f,t[e+1]=c*g+h*f+l*d-s*p,t[e+2]=l*g+h*p+s*f-c*d,t[e+3]=h*g-s*d-c*f-l*p,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,i){return this._x=t,this._y=e,this._z=n,this._w=i,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e){const n=t._x,i=t._y,r=t._z,o=t._order,s=Math.cos,c=Math.sin,l=s(n/2),h=s(i/2),d=s(r/2),f=c(n/2),p=c(i/2),g=c(r/2);switch(o){case"XYZ":this._x=f*h*d+l*p*g,this._y=l*p*d-f*h*g,this._z=l*h*g+f*p*d,this._w=l*h*d-f*p*g;break;case"YXZ":this._x=f*h*d+l*p*g,this._y=l*p*d-f*h*g,this._z=l*h*g-f*p*d,this._w=l*h*d+f*p*g;break;case"ZXY":this._x=f*h*d-l*p*g,this._y=l*p*d+f*h*g,this._z=l*h*g+f*p*d,this._w=l*h*d-f*p*g;break;case"ZYX":this._x=f*h*d-l*p*g,this._y=l*p*d+f*h*g,this._z=l*h*g-f*p*d,this._w=l*h*d+f*p*g;break;case"YZX":this._x=f*h*d+l*p*g,this._y=l*p*d+f*h*g,this._z=l*h*g-f*p*d,this._w=l*h*d-f*p*g;break;case"XZY":this._x=f*h*d-l*p*g,this._y=l*p*d-f*h*g,this._z=l*h*g+f*p*d,this._w=l*h*d+f*p*g;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+o)}return e!==!1&&this._onChangeCallback(),this}setFromAxisAngle(t,e){const n=e/2,i=Math.sin(n);return this._x=t.x*i,this._y=t.y*i,this._z=t.z*i,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){const e=t.elements,n=e[0],i=e[4],r=e[8],o=e[1],s=e[5],c=e[9],l=e[2],h=e[6],d=e[10],f=n+s+d;if(f>0){const p=.5/Math.sqrt(f+1);this._w=.25/p,this._x=(h-c)*p,this._y=(r-l)*p,this._z=(o-i)*p}else if(n>s&&n>d){const p=2*Math.sqrt(1+n-s-d);this._w=(h-c)/p,this._x=.25*p,this._y=(i+o)/p,this._z=(r+l)/p}else if(s>d){const p=2*Math.sqrt(1+s-n-d);this._w=(r-l)/p,this._x=(i+o)/p,this._y=.25*p,this._z=(c+h)/p}else{const p=2*Math.sqrt(1+d-n-s);this._w=(o-i)/p,this._x=(r+l)/p,this._y=(c+h)/p,this._z=.25*p}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<Number.EPSILON?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(ne(this.dot(t),-1,1)))}rotateTowards(t,e){const n=this.angleTo(t);if(n===0)return this;const i=Math.min(1,e/n);return this.slerp(t,i),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){const n=t._x,i=t._y,r=t._z,o=t._w,s=e._x,c=e._y,l=e._z,h=e._w;return this._x=n*h+o*s+i*l-r*c,this._y=i*h+o*c+r*s-n*l,this._z=r*h+o*l+n*c-i*s,this._w=o*h-n*s-i*c-r*l,this._onChangeCallback(),this}slerp(t,e){if(e===0)return this;if(e===1)return this.copy(t);const n=this._x,i=this._y,r=this._z,o=this._w;let s=o*t._w+n*t._x+i*t._y+r*t._z;if(s<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,s=-s):this.copy(t),s>=1)return this._w=o,this._x=n,this._y=i,this._z=r,this;const c=1-s*s;if(c<=Number.EPSILON){const p=1-e;return this._w=p*o+e*this._w,this._x=p*n+e*this._x,this._y=p*i+e*this._y,this._z=p*r+e*this._z,this.normalize(),this._onChangeCallback(),this}const l=Math.sqrt(c),h=Math.atan2(l,s),d=Math.sin((1-e)*h)/l,f=Math.sin(e*h)/l;return this._w=o*d+this._w*f,this._x=n*d+this._x*f,this._y=i*d+this._y*f,this._z=r*d+this._z*f,this._onChangeCallback(),this}slerpQuaternions(t,e,n){return this.copy(t).slerp(e,n)}random(){const t=Math.random(),e=Math.sqrt(1-t),n=Math.sqrt(t),i=2*Math.PI*Math.random(),r=2*Math.PI*Math.random();return this.set(e*Math.cos(i),n*Math.sin(r),n*Math.cos(r),e*Math.sin(i))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class P{constructor(t=0,e=0,n=0){P.prototype.isVector3=!0,this.x=t,this.y=e,this.z=n}set(t,e,n){return n===void 0&&(n=this.z),this.x=t,this.y=e,this.z=n,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return this.applyQuaternion(Ls.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(Ls.setFromAxisAngle(t,e))}applyMatrix3(t){const e=this.x,n=this.y,i=this.z,r=t.elements;return this.x=r[0]*e+r[3]*n+r[6]*i,this.y=r[1]*e+r[4]*n+r[7]*i,this.z=r[2]*e+r[5]*n+r[8]*i,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){const e=this.x,n=this.y,i=this.z,r=t.elements,o=1/(r[3]*e+r[7]*n+r[11]*i+r[15]);return this.x=(r[0]*e+r[4]*n+r[8]*i+r[12])*o,this.y=(r[1]*e+r[5]*n+r[9]*i+r[13])*o,this.z=(r[2]*e+r[6]*n+r[10]*i+r[14])*o,this}applyQuaternion(t){const e=this.x,n=this.y,i=this.z,r=t.x,o=t.y,s=t.z,c=t.w,l=c*e+o*i-s*n,h=c*n+s*e-r*i,d=c*i+r*n-o*e,f=-r*e-o*n-s*i;return this.x=l*c+f*-r+h*-s-d*-o,this.y=h*c+f*-o+d*-r-l*-s,this.z=d*c+f*-s+l*-o-h*-r,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){const e=this.x,n=this.y,i=this.z,r=t.elements;return this.x=r[0]*e+r[4]*n+r[8]*i,this.y=r[1]*e+r[5]*n+r[9]*i,this.z=r[2]*e+r[6]*n+r[10]*i,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this.z=this.z<0?Math.ceil(this.z):Math.floor(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this}cross(t){return this.crossVectors(this,t)}crossVectors(t,e){const n=t.x,i=t.y,r=t.z,o=e.x,s=e.y,c=e.z;return this.x=i*c-r*s,this.y=r*o-n*c,this.z=n*s-i*o,this}projectOnVector(t){const e=t.lengthSq();if(e===0)return this.set(0,0,0);const n=t.dot(this)/e;return this.copy(t).multiplyScalar(n)}projectOnPlane(t){return nr.copy(this).projectOnVector(t),this.sub(nr)}reflect(t){return this.sub(nr.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(ne(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y,i=this.z-t.z;return e*e+n*n+i*i}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,n){const i=Math.sin(e)*t;return this.x=i*Math.sin(n),this.y=Math.cos(e)*t,this.z=i*Math.cos(n),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,n){return this.x=t*Math.sin(e),this.y=n,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){const e=this.setFromMatrixColumn(t,0).length(),n=this.setFromMatrixColumn(t,1).length(),i=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=n,this.z=i,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,e*4)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,e*3)}setFromEuler(t){return this.x=t._x,this.y=t._y,this.z=t._z,this}setFromColor(t){return this.x=t.r,this.y=t.g,this.z=t.b,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const t=(Math.random()-.5)*2,e=Math.random()*Math.PI*2,n=Math.sqrt(1-t**2);return this.x=n*Math.cos(e),this.y=n*Math.sin(e),this.z=t,this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const nr=new P,Ls=new vn;class ci{constructor(t=new P(1/0,1/0,1/0),e=new P(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e+=3)this.expandByPoint(Ve.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,n=t.count;e<n;e++)this.expandByPoint(Ve.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=Ve.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){if(t.updateWorldMatrix(!1,!1),t.boundingBox!==void 0)t.boundingBox===null&&t.computeBoundingBox(),wn.copy(t.boundingBox),wn.applyMatrix4(t.matrixWorld),this.union(wn);else{const i=t.geometry;if(i!==void 0)if(e&&i.attributes!==void 0&&i.attributes.position!==void 0){const r=i.attributes.position;for(let o=0,s=r.count;o<s;o++)Ve.fromBufferAttribute(r,o).applyMatrix4(t.matrixWorld),this.expandByPoint(Ve)}else i.boundingBox===null&&i.computeBoundingBox(),wn.copy(i.boundingBox),wn.applyMatrix4(t.matrixWorld),this.union(wn)}const n=t.children;for(let i=0,r=n.length;i<r;i++)this.expandByObject(n[i],e);return this}containsPoint(t){return!(t.x<this.min.x||t.x>this.max.x||t.y<this.min.y||t.y>this.max.y||t.z<this.min.z||t.z>this.max.z)}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return!(t.max.x<this.min.x||t.min.x>this.max.x||t.max.y<this.min.y||t.min.y>this.max.y||t.max.z<this.min.z||t.min.z>this.max.z)}intersectsSphere(t){return this.clampPoint(t.center,Ve),Ve.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(Kn),fi.subVectors(this.max,Kn),En.subVectors(t.a,Kn),Tn.subVectors(t.b,Kn),An.subVectors(t.c,Kn),Ze.subVectors(Tn,En),Ke.subVectors(An,Tn),an.subVectors(En,An);let e=[0,-Ze.z,Ze.y,0,-Ke.z,Ke.y,0,-an.z,an.y,Ze.z,0,-Ze.x,Ke.z,0,-Ke.x,an.z,0,-an.x,-Ze.y,Ze.x,0,-Ke.y,Ke.x,0,-an.y,an.x,0];return!ir(e,En,Tn,An,fi)||(e=[1,0,0,0,1,0,0,0,1],!ir(e,En,Tn,An,fi))?!1:(pi.crossVectors(Ze,Ke),e=[pi.x,pi.y,pi.z],ir(e,En,Tn,An,fi))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,Ve).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(Ve).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(Ge[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Ge[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Ge[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Ge[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Ge[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Ge[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Ge[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Ge[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Ge),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}}const Ge=[new P,new P,new P,new P,new P,new P,new P,new P],Ve=new P,wn=new ci,En=new P,Tn=new P,An=new P,Ze=new P,Ke=new P,an=new P,Kn=new P,fi=new P,pi=new P,on=new P;function ir(a,t,e,n,i){for(let r=0,o=a.length-3;r<=o;r+=3){on.fromArray(a,r);const s=i.x*Math.abs(on.x)+i.y*Math.abs(on.y)+i.z*Math.abs(on.z),c=t.dot(on),l=e.dot(on),h=n.dot(on);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>s)return!1}return!0}const al=new ci,$n=new P,rr=new P;class Nr{constructor(t=new P,e=-1){this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){const n=this.center;e!==void 0?n.copy(e):al.setFromPoints(t).getCenter(n);let i=0;for(let r=0,o=t.length;r<o;r++)i=Math.max(i,n.distanceToSquared(t[r]));return this.radius=Math.sqrt(i),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){const e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){const n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;$n.subVectors(t,this.center);const e=$n.lengthSq();if(e>this.radius*this.radius){const n=Math.sqrt(e),i=(n-this.radius)*.5;this.center.addScaledVector($n,i/n),this.radius+=i}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(rr.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint($n.copy(t.center).add(rr)),this.expandByPoint($n.copy(t.center).sub(rr))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}}const ke=new P,sr=new P,mi=new P,$e=new P,ar=new P,gi=new P,or=new P;class Ia{constructor(t=new P,e=new P(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,ke)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);const n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){const e=ke.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(ke.copy(this.origin).addScaledVector(this.direction,e),ke.distanceToSquared(t))}distanceSqToSegment(t,e,n,i){sr.copy(t).add(e).multiplyScalar(.5),mi.copy(e).sub(t).normalize(),$e.copy(this.origin).sub(sr);const r=t.distanceTo(e)*.5,o=-this.direction.dot(mi),s=$e.dot(this.direction),c=-$e.dot(mi),l=$e.lengthSq(),h=Math.abs(1-o*o);let d,f,p,g;if(h>0)if(d=o*c-s,f=o*s-c,g=r*h,d>=0)if(f>=-g)if(f<=g){const _=1/h;d*=_,f*=_,p=d*(d+o*f+2*s)+f*(o*d+f+2*c)+l}else f=r,d=Math.max(0,-(o*f+s)),p=-d*d+f*(f+2*c)+l;else f=-r,d=Math.max(0,-(o*f+s)),p=-d*d+f*(f+2*c)+l;else f<=-g?(d=Math.max(0,-(-o*r+s)),f=d>0?-r:Math.min(Math.max(-r,-c),r),p=-d*d+f*(f+2*c)+l):f<=g?(d=0,f=Math.min(Math.max(-r,-c),r),p=f*(f+2*c)+l):(d=Math.max(0,-(o*r+s)),f=d>0?r:Math.min(Math.max(-r,-c),r),p=-d*d+f*(f+2*c)+l);else f=o>0?-r:r,d=Math.max(0,-(o*f+s)),p=-d*d+f*(f+2*c)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,d),i&&i.copy(sr).addScaledVector(mi,f),p}intersectSphere(t,e){ke.subVectors(t.center,this.origin);const n=ke.dot(this.direction),i=ke.dot(ke)-n*n,r=t.radius*t.radius;if(i>r)return null;const o=Math.sqrt(r-i),s=n-o,c=n+o;return c<0?null:s<0?this.at(c,e):this.at(s,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){const e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){const n=this.distanceToPlane(t);return n===null?null:this.at(n,e)}intersectsPlane(t){const e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,i,r,o,s,c;const l=1/this.direction.x,h=1/this.direction.y,d=1/this.direction.z,f=this.origin;return l>=0?(n=(t.min.x-f.x)*l,i=(t.max.x-f.x)*l):(n=(t.max.x-f.x)*l,i=(t.min.x-f.x)*l),h>=0?(r=(t.min.y-f.y)*h,o=(t.max.y-f.y)*h):(r=(t.max.y-f.y)*h,o=(t.min.y-f.y)*h),n>o||r>i||((r>n||isNaN(n))&&(n=r),(o<i||isNaN(i))&&(i=o),d>=0?(s=(t.min.z-f.z)*d,c=(t.max.z-f.z)*d):(s=(t.max.z-f.z)*d,c=(t.min.z-f.z)*d),n>c||s>i)||((s>n||n!==n)&&(n=s),(c<i||i!==i)&&(i=c),i<0)?null:this.at(n>=0?n:i,e)}intersectsBox(t){return this.intersectBox(t,ke)!==null}intersectTriangle(t,e,n,i,r){ar.subVectors(e,t),gi.subVectors(n,t),or.crossVectors(ar,gi);let o=this.direction.dot(or),s;if(o>0){if(i)return null;s=1}else if(o<0)s=-1,o=-o;else return null;$e.subVectors(this.origin,t);const c=s*this.direction.dot(gi.crossVectors($e,gi));if(c<0)return null;const l=s*this.direction.dot(ar.cross($e));if(l<0||c+l>o)return null;const h=-s*$e.dot(or);return h<0?null:this.at(h/o,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class Kt{constructor(){Kt.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}set(t,e,n,i,r,o,s,c,l,h,d,f,p,g,_,m){const u=this.elements;return u[0]=t,u[4]=e,u[8]=n,u[12]=i,u[1]=r,u[5]=o,u[9]=s,u[13]=c,u[2]=l,u[6]=h,u[10]=d,u[14]=f,u[3]=p,u[7]=g,u[11]=_,u[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Kt().fromArray(this.elements)}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],this}copyPosition(t){const e=this.elements,n=t.elements;return e[12]=n[12],e[13]=n[13],e[14]=n[14],this}setFromMatrix3(t){const e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,n){return t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(t,e,n){return this.set(t.x,e.x,n.x,0,t.y,e.y,n.y,0,t.z,e.z,n.z,0,0,0,0,1),this}extractRotation(t){const e=this.elements,n=t.elements,i=1/Cn.setFromMatrixColumn(t,0).length(),r=1/Cn.setFromMatrixColumn(t,1).length(),o=1/Cn.setFromMatrixColumn(t,2).length();return e[0]=n[0]*i,e[1]=n[1]*i,e[2]=n[2]*i,e[3]=0,e[4]=n[4]*r,e[5]=n[5]*r,e[6]=n[6]*r,e[7]=0,e[8]=n[8]*o,e[9]=n[9]*o,e[10]=n[10]*o,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){const e=this.elements,n=t.x,i=t.y,r=t.z,o=Math.cos(n),s=Math.sin(n),c=Math.cos(i),l=Math.sin(i),h=Math.cos(r),d=Math.sin(r);if(t.order==="XYZ"){const f=o*h,p=o*d,g=s*h,_=s*d;e[0]=c*h,e[4]=-c*d,e[8]=l,e[1]=p+g*l,e[5]=f-_*l,e[9]=-s*c,e[2]=_-f*l,e[6]=g+p*l,e[10]=o*c}else if(t.order==="YXZ"){const f=c*h,p=c*d,g=l*h,_=l*d;e[0]=f+_*s,e[4]=g*s-p,e[8]=o*l,e[1]=o*d,e[5]=o*h,e[9]=-s,e[2]=p*s-g,e[6]=_+f*s,e[10]=o*c}else if(t.order==="ZXY"){const f=c*h,p=c*d,g=l*h,_=l*d;e[0]=f-_*s,e[4]=-o*d,e[8]=g+p*s,e[1]=p+g*s,e[5]=o*h,e[9]=_-f*s,e[2]=-o*l,e[6]=s,e[10]=o*c}else if(t.order==="ZYX"){const f=o*h,p=o*d,g=s*h,_=s*d;e[0]=c*h,e[4]=g*l-p,e[8]=f*l+_,e[1]=c*d,e[5]=_*l+f,e[9]=p*l-g,e[2]=-l,e[6]=s*c,e[10]=o*c}else if(t.order==="YZX"){const f=o*c,p=o*l,g=s*c,_=s*l;e[0]=c*h,e[4]=_-f*d,e[8]=g*d+p,e[1]=d,e[5]=o*h,e[9]=-s*h,e[2]=-l*h,e[6]=p*d+g,e[10]=f-_*d}else if(t.order==="XZY"){const f=o*c,p=o*l,g=s*c,_=s*l;e[0]=c*h,e[4]=-d,e[8]=l*h,e[1]=f*d+_,e[5]=o*h,e[9]=p*d-g,e[2]=g*d-p,e[6]=s*h,e[10]=_*d+f}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(ol,t,ll)}lookAt(t,e,n){const i=this.elements;return ve.subVectors(t,e),ve.lengthSq()===0&&(ve.z=1),ve.normalize(),Qe.crossVectors(n,ve),Qe.lengthSq()===0&&(Math.abs(n.z)===1?ve.x+=1e-4:ve.z+=1e-4,ve.normalize(),Qe.crossVectors(n,ve)),Qe.normalize(),_i.crossVectors(ve,Qe),i[0]=Qe.x,i[4]=_i.x,i[8]=ve.x,i[1]=Qe.y,i[5]=_i.y,i[9]=ve.y,i[2]=Qe.z,i[6]=_i.z,i[10]=ve.z,this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,i=e.elements,r=this.elements,o=n[0],s=n[4],c=n[8],l=n[12],h=n[1],d=n[5],f=n[9],p=n[13],g=n[2],_=n[6],m=n[10],u=n[14],E=n[3],x=n[7],b=n[11],y=n[15],C=i[0],D=i[4],R=i[8],M=i[12],T=i[1],X=i[5],q=i[9],I=i[13],N=i[2],B=i[6],Q=i[10],et=i[14],j=i[3],Z=i[7],nt=i[11],vt=i[15];return r[0]=o*C+s*T+c*N+l*j,r[4]=o*D+s*X+c*B+l*Z,r[8]=o*R+s*q+c*Q+l*nt,r[12]=o*M+s*I+c*et+l*vt,r[1]=h*C+d*T+f*N+p*j,r[5]=h*D+d*X+f*B+p*Z,r[9]=h*R+d*q+f*Q+p*nt,r[13]=h*M+d*I+f*et+p*vt,r[2]=g*C+_*T+m*N+u*j,r[6]=g*D+_*X+m*B+u*Z,r[10]=g*R+_*q+m*Q+u*nt,r[14]=g*M+_*I+m*et+u*vt,r[3]=E*C+x*T+b*N+y*j,r[7]=E*D+x*X+b*B+y*Z,r[11]=E*R+x*q+b*Q+y*nt,r[15]=E*M+x*I+b*et+y*vt,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[4],i=t[8],r=t[12],o=t[1],s=t[5],c=t[9],l=t[13],h=t[2],d=t[6],f=t[10],p=t[14],g=t[3],_=t[7],m=t[11],u=t[15];return g*(+r*c*d-i*l*d-r*s*f+n*l*f+i*s*p-n*c*p)+_*(+e*c*p-e*l*f+r*o*f-i*o*p+i*l*h-r*c*h)+m*(+e*l*d-e*s*p-r*o*d+n*o*p+r*s*h-n*l*h)+u*(-i*s*h-e*c*d+e*s*f+i*o*d-n*o*f+n*c*h)}transpose(){const t=this.elements;let e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,n){const i=this.elements;return t.isVector3?(i[12]=t.x,i[13]=t.y,i[14]=t.z):(i[12]=t,i[13]=e,i[14]=n),this}invert(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],o=t[4],s=t[5],c=t[6],l=t[7],h=t[8],d=t[9],f=t[10],p=t[11],g=t[12],_=t[13],m=t[14],u=t[15],E=d*m*l-_*f*l+_*c*p-s*m*p-d*c*u+s*f*u,x=g*f*l-h*m*l-g*c*p+o*m*p+h*c*u-o*f*u,b=h*_*l-g*d*l+g*s*p-o*_*p-h*s*u+o*d*u,y=g*d*c-h*_*c-g*s*f+o*_*f+h*s*m-o*d*m,C=e*E+n*x+i*b+r*y;if(C===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const D=1/C;return t[0]=E*D,t[1]=(_*f*r-d*m*r-_*i*p+n*m*p+d*i*u-n*f*u)*D,t[2]=(s*m*r-_*c*r+_*i*l-n*m*l-s*i*u+n*c*u)*D,t[3]=(d*c*r-s*f*r-d*i*l+n*f*l+s*i*p-n*c*p)*D,t[4]=x*D,t[5]=(h*m*r-g*f*r+g*i*p-e*m*p-h*i*u+e*f*u)*D,t[6]=(g*c*r-o*m*r-g*i*l+e*m*l+o*i*u-e*c*u)*D,t[7]=(o*f*r-h*c*r+h*i*l-e*f*l-o*i*p+e*c*p)*D,t[8]=b*D,t[9]=(g*d*r-h*_*r-g*n*p+e*_*p+h*n*u-e*d*u)*D,t[10]=(o*_*r-g*s*r+g*n*l-e*_*l-o*n*u+e*s*u)*D,t[11]=(h*s*r-o*d*r-h*n*l+e*d*l+o*n*p-e*s*p)*D,t[12]=y*D,t[13]=(h*_*i-g*d*i+g*n*f-e*_*f-h*n*m+e*d*m)*D,t[14]=(g*s*i-o*_*i-g*n*c+e*_*c+o*n*m-e*s*m)*D,t[15]=(o*d*i-h*s*i+h*n*c-e*d*c-o*n*f+e*s*f)*D,this}scale(t){const e=this.elements,n=t.x,i=t.y,r=t.z;return e[0]*=n,e[4]*=i,e[8]*=r,e[1]*=n,e[5]*=i,e[9]*=r,e[2]*=n,e[6]*=i,e[10]*=r,e[3]*=n,e[7]*=i,e[11]*=r,this}getMaxScaleOnAxis(){const t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],i=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,n,i))}makeTranslation(t,e,n){return this.set(1,0,0,t,0,1,0,e,0,0,1,n,0,0,0,1),this}makeRotationX(t){const e=Math.cos(t),n=Math.sin(t);return this.set(1,0,0,0,0,e,-n,0,0,n,e,0,0,0,0,1),this}makeRotationY(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,0,n,0,0,1,0,0,-n,0,e,0,0,0,0,1),this}makeRotationZ(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,0,n,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){const n=Math.cos(e),i=Math.sin(e),r=1-n,o=t.x,s=t.y,c=t.z,l=r*o,h=r*s;return this.set(l*o+n,l*s-i*c,l*c+i*s,0,l*s+i*c,h*s+n,h*c-i*o,0,l*c-i*s,h*c+i*o,r*c*c+n,0,0,0,0,1),this}makeScale(t,e,n){return this.set(t,0,0,0,0,e,0,0,0,0,n,0,0,0,0,1),this}makeShear(t,e,n,i,r,o){return this.set(1,n,r,0,t,1,o,0,e,i,1,0,0,0,0,1),this}compose(t,e,n){const i=this.elements,r=e._x,o=e._y,s=e._z,c=e._w,l=r+r,h=o+o,d=s+s,f=r*l,p=r*h,g=r*d,_=o*h,m=o*d,u=s*d,E=c*l,x=c*h,b=c*d,y=n.x,C=n.y,D=n.z;return i[0]=(1-(_+u))*y,i[1]=(p+b)*y,i[2]=(g-x)*y,i[3]=0,i[4]=(p-b)*C,i[5]=(1-(f+u))*C,i[6]=(m+E)*C,i[7]=0,i[8]=(g+x)*D,i[9]=(m-E)*D,i[10]=(1-(f+_))*D,i[11]=0,i[12]=t.x,i[13]=t.y,i[14]=t.z,i[15]=1,this}decompose(t,e,n){const i=this.elements;let r=Cn.set(i[0],i[1],i[2]).length();const o=Cn.set(i[4],i[5],i[6]).length(),s=Cn.set(i[8],i[9],i[10]).length();this.determinant()<0&&(r=-r),t.x=i[12],t.y=i[13],t.z=i[14],Ae.copy(this);const l=1/r,h=1/o,d=1/s;return Ae.elements[0]*=l,Ae.elements[1]*=l,Ae.elements[2]*=l,Ae.elements[4]*=h,Ae.elements[5]*=h,Ae.elements[6]*=h,Ae.elements[8]*=d,Ae.elements[9]*=d,Ae.elements[10]*=d,e.setFromRotationMatrix(Ae),n.x=r,n.y=o,n.z=s,this}makePerspective(t,e,n,i,r,o){const s=this.elements,c=2*r/(e-t),l=2*r/(n-i),h=(e+t)/(e-t),d=(n+i)/(n-i),f=-(o+r)/(o-r),p=-2*o*r/(o-r);return s[0]=c,s[4]=0,s[8]=h,s[12]=0,s[1]=0,s[5]=l,s[9]=d,s[13]=0,s[2]=0,s[6]=0,s[10]=f,s[14]=p,s[3]=0,s[7]=0,s[11]=-1,s[15]=0,this}makeOrthographic(t,e,n,i,r,o){const s=this.elements,c=1/(e-t),l=1/(n-i),h=1/(o-r),d=(e+t)*c,f=(n+i)*l,p=(o+r)*h;return s[0]=2*c,s[4]=0,s[8]=0,s[12]=-d,s[1]=0,s[5]=2*l,s[9]=0,s[13]=-f,s[2]=0,s[6]=0,s[10]=-2*h,s[14]=-p,s[3]=0,s[7]=0,s[11]=0,s[15]=1,this}equals(t){const e=this.elements,n=t.elements;for(let i=0;i<16;i++)if(e[i]!==n[i])return!1;return!0}fromArray(t,e=0){for(let n=0;n<16;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t[e+9]=n[9],t[e+10]=n[10],t[e+11]=n[11],t[e+12]=n[12],t[e+13]=n[13],t[e+14]=n[14],t[e+15]=n[15],t}}const Cn=new P,Ae=new Kt,ol=new P(0,0,0),ll=new P(1,1,1),Qe=new P,_i=new P,ve=new P,Ps=new Kt,Rs=new vn;class Fi{constructor(t=0,e=0,n=0,i=Fi.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=e,this._z=n,this._order=i}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,n,i=this._order){return this._x=t,this._y=e,this._z=n,this._order=i,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,n=!0){const i=t.elements,r=i[0],o=i[4],s=i[8],c=i[1],l=i[5],h=i[9],d=i[2],f=i[6],p=i[10];switch(e){case"XYZ":this._y=Math.asin(ne(s,-1,1)),Math.abs(s)<.9999999?(this._x=Math.atan2(-h,p),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(f,l),this._z=0);break;case"YXZ":this._x=Math.asin(-ne(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(s,p),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-d,r),this._z=0);break;case"ZXY":this._x=Math.asin(ne(f,-1,1)),Math.abs(f)<.9999999?(this._y=Math.atan2(-d,p),this._z=Math.atan2(-o,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-ne(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(f,p),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-o,l));break;case"YZX":this._z=Math.asin(ne(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-d,r)):(this._x=0,this._y=Math.atan2(s,p));break;case"XZY":this._z=Math.asin(-ne(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(f,l),this._y=Math.atan2(s,r)):(this._x=Math.atan2(-h,p),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,n===!0&&this._onChangeCallback(),this}setFromQuaternion(t,e,n){return Ps.makeRotationFromQuaternion(t),this.setFromRotationMatrix(Ps,e,n)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return Rs.setFromEuler(this),this.setFromQuaternion(Rs,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],t[3]!==void 0&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Fi.DEFAULT_ORDER="XYZ";class Or{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}}let cl=0;const Ds=new P,Ln=new vn,He=new Kt,vi=new P,Qn=new P,hl=new P,ul=new vn,Is=new P(1,0,0),Us=new P(0,1,0),Ns=new P(0,0,1),dl={type:"added"},Os={type:"removed"};class re extends Mn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:cl++}),this.uuid=li(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=re.DEFAULT_UP.clone();const t=new P,e=new Fi,n=new vn,i=new P(1,1,1);function r(){n.setFromEuler(e,!1)}function o(){e.setFromQuaternion(n,void 0,!1)}e._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:i},modelViewMatrix:{value:new Kt},normalMatrix:{value:new It}}),this.matrix=new Kt,this.matrixWorld=new Kt,this.matrixAutoUpdate=re.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.matrixWorldAutoUpdate=re.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.layers=new Or,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return Ln.setFromAxisAngle(t,e),this.quaternion.multiply(Ln),this}rotateOnWorldAxis(t,e){return Ln.setFromAxisAngle(t,e),this.quaternion.premultiply(Ln),this}rotateX(t){return this.rotateOnAxis(Is,t)}rotateY(t){return this.rotateOnAxis(Us,t)}rotateZ(t){return this.rotateOnAxis(Ns,t)}translateOnAxis(t,e){return Ds.copy(t).applyQuaternion(this.quaternion),this.position.add(Ds.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(Is,t)}translateY(t){return this.translateOnAxis(Us,t)}translateZ(t){return this.translateOnAxis(Ns,t)}localToWorld(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(He.copy(this.matrixWorld).invert())}lookAt(t,e,n){t.isVector3?vi.copy(t):vi.set(t,e,n);const i=this.parent;this.updateWorldMatrix(!0,!1),Qn.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?He.lookAt(Qn,vi,this.up):He.lookAt(vi,Qn,this.up),this.quaternion.setFromRotationMatrix(He),i&&(He.extractRotation(i.matrixWorld),Ln.setFromRotationMatrix(He),this.quaternion.premultiply(Ln.invert()))}add(t){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return t===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(t.parent!==null&&t.parent.remove(t),t.parent=this,this.children.push(t),t.dispatchEvent(dl)):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const e=this.children.indexOf(t);return e!==-1&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(Os)),this}removeFromParent(){const t=this.parent;return t!==null&&t.remove(this),this}clear(){for(let t=0;t<this.children.length;t++){const e=this.children[t];e.parent=null,e.dispatchEvent(Os)}return this.children.length=0,this}attach(t){return this.updateWorldMatrix(!0,!1),He.copy(this.matrixWorld).invert(),t.parent!==null&&(t.parent.updateWorldMatrix(!0,!1),He.multiply(t.parent.matrixWorld)),t.applyMatrix4(He),this.add(t),t.updateWorldMatrix(!1,!0),this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let n=0,i=this.children.length;n<i;n++){const o=this.children[n].getObjectByProperty(t,e);if(o!==void 0)return o}}getObjectsByProperty(t,e){let n=[];this[t]===e&&n.push(this);for(let i=0,r=this.children.length;i<r;i++){const o=this.children[i].getObjectsByProperty(t,e);o.length>0&&(n=n.concat(o))}return n}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Qn,t,hl),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Qn,ul,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(t){t(this);const e=this.children;for(let n=0,i=e.length;n<i;n++)e[n].traverse(t)}traverseVisible(t){if(this.visible===!1)return;t(this);const e=this.children;for(let n=0,i=e.length;n<i;n++)e[n].traverseVisible(t)}traverseAncestors(t){const e=this.parent;e!==null&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),this.matrixWorldNeedsUpdate=!1,t=!0);const e=this.children;for(let n=0,i=e.length;n<i;n++){const r=e[n];(r.matrixWorldAutoUpdate===!0||t===!0)&&r.updateMatrixWorld(t)}}updateWorldMatrix(t,e){const n=this.parent;if(t===!0&&n!==null&&n.matrixWorldAutoUpdate===!0&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),e===!0){const i=this.children;for(let r=0,o=i.length;r<o;r++){const s=i[r];s.matrixWorldAutoUpdate===!0&&s.updateWorldMatrix(!1,!0)}}}toJSON(t){const e=t===void 0||typeof t=="string",n={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.5,type:"Object",generator:"Object3D.toJSON"});const i={};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.castShadow===!0&&(i.castShadow=!0),this.receiveShadow===!0&&(i.receiveShadow=!0),this.visible===!1&&(i.visible=!1),this.frustumCulled===!1&&(i.frustumCulled=!1),this.renderOrder!==0&&(i.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(i.userData=this.userData),i.layers=this.layers.mask,i.matrix=this.matrix.toArray(),i.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(i.matrixAutoUpdate=!1),this.isInstancedMesh&&(i.type="InstancedMesh",i.count=this.count,i.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(i.instanceColor=this.instanceColor.toJSON()));function r(s,c){return s[c.uuid]===void 0&&(s[c.uuid]=c.toJSON(t)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?i.background=this.background.toJSON():this.background.isTexture&&(i.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(i.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){i.geometry=r(t.geometries,this.geometry);const s=this.geometry.parameters;if(s!==void 0&&s.shapes!==void 0){const c=s.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){const d=c[l];r(t.shapes,d)}else r(t.shapes,c)}}if(this.isSkinnedMesh&&(i.bindMode=this.bindMode,i.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(t.skeletons,this.skeleton),i.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const s=[];for(let c=0,l=this.material.length;c<l;c++)s.push(r(t.materials,this.material[c]));i.material=s}else i.material=r(t.materials,this.material);if(this.children.length>0){i.children=[];for(let s=0;s<this.children.length;s++)i.children.push(this.children[s].toJSON(t).object)}if(this.animations.length>0){i.animations=[];for(let s=0;s<this.animations.length;s++){const c=this.animations[s];i.animations.push(r(t.animations,c))}}if(e){const s=o(t.geometries),c=o(t.materials),l=o(t.textures),h=o(t.images),d=o(t.shapes),f=o(t.skeletons),p=o(t.animations),g=o(t.nodes);s.length>0&&(n.geometries=s),c.length>0&&(n.materials=c),l.length>0&&(n.textures=l),h.length>0&&(n.images=h),d.length>0&&(n.shapes=d),f.length>0&&(n.skeletons=f),p.length>0&&(n.animations=p),g.length>0&&(n.nodes=g)}return n.object=i,n;function o(s){const c=[];for(const l in s){const h=s[l];delete h.metadata,c.push(h)}return c}}clone(t){return new this.constructor().copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.matrixWorldAutoUpdate=t.matrixWorldAutoUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.userData=JSON.parse(JSON.stringify(t.userData)),e===!0)for(let n=0;n<t.children.length;n++){const i=t.children[n];this.add(i.clone())}return this}}re.DEFAULT_UP=new P(0,1,0);re.DEFAULT_MATRIX_AUTO_UPDATE=!0;re.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Ce=new P,We=new P,lr=new P,qe=new P,Pn=new P,Rn=new P,Fs=new P,cr=new P,hr=new P,ur=new P;let xi=!1;class Pe{constructor(t=new P,e=new P,n=new P){this.a=t,this.b=e,this.c=n}static getNormal(t,e,n,i){i.subVectors(n,e),Ce.subVectors(t,e),i.cross(Ce);const r=i.lengthSq();return r>0?i.multiplyScalar(1/Math.sqrt(r)):i.set(0,0,0)}static getBarycoord(t,e,n,i,r){Ce.subVectors(i,e),We.subVectors(n,e),lr.subVectors(t,e);const o=Ce.dot(Ce),s=Ce.dot(We),c=Ce.dot(lr),l=We.dot(We),h=We.dot(lr),d=o*l-s*s;if(d===0)return r.set(-2,-1,-1);const f=1/d,p=(l*c-s*h)*f,g=(o*h-s*c)*f;return r.set(1-p-g,g,p)}static containsPoint(t,e,n,i){return this.getBarycoord(t,e,n,i,qe),qe.x>=0&&qe.y>=0&&qe.x+qe.y<=1}static getUV(t,e,n,i,r,o,s,c){return xi===!1&&(console.warn("THREE.Triangle.getUV() has been renamed to THREE.Triangle.getInterpolation()."),xi=!0),this.getInterpolation(t,e,n,i,r,o,s,c)}static getInterpolation(t,e,n,i,r,o,s,c){return this.getBarycoord(t,e,n,i,qe),c.setScalar(0),c.addScaledVector(r,qe.x),c.addScaledVector(o,qe.y),c.addScaledVector(s,qe.z),c}static isFrontFacing(t,e,n,i){return Ce.subVectors(n,e),We.subVectors(t,e),Ce.cross(We).dot(i)<0}set(t,e,n){return this.a.copy(t),this.b.copy(e),this.c.copy(n),this}setFromPointsAndIndices(t,e,n,i){return this.a.copy(t[e]),this.b.copy(t[n]),this.c.copy(t[i]),this}setFromAttributeAndIndices(t,e,n,i){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,n),this.c.fromBufferAttribute(t,i),this}clone(){return new this.constructor().copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return Ce.subVectors(this.c,this.b),We.subVectors(this.a,this.b),Ce.cross(We).length()*.5}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return Pe.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return Pe.getBarycoord(t,this.a,this.b,this.c,e)}getUV(t,e,n,i,r){return xi===!1&&(console.warn("THREE.Triangle.getUV() has been renamed to THREE.Triangle.getInterpolation()."),xi=!0),Pe.getInterpolation(t,this.a,this.b,this.c,e,n,i,r)}getInterpolation(t,e,n,i,r){return Pe.getInterpolation(t,this.a,this.b,this.c,e,n,i,r)}containsPoint(t){return Pe.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return Pe.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){const n=this.a,i=this.b,r=this.c;let o,s;Pn.subVectors(i,n),Rn.subVectors(r,n),cr.subVectors(t,n);const c=Pn.dot(cr),l=Rn.dot(cr);if(c<=0&&l<=0)return e.copy(n);hr.subVectors(t,i);const h=Pn.dot(hr),d=Rn.dot(hr);if(h>=0&&d<=h)return e.copy(i);const f=c*d-h*l;if(f<=0&&c>=0&&h<=0)return o=c/(c-h),e.copy(n).addScaledVector(Pn,o);ur.subVectors(t,r);const p=Pn.dot(ur),g=Rn.dot(ur);if(g>=0&&p<=g)return e.copy(r);const _=p*l-c*g;if(_<=0&&l>=0&&g<=0)return s=l/(l-g),e.copy(n).addScaledVector(Rn,s);const m=h*g-p*d;if(m<=0&&d-h>=0&&p-g>=0)return Fs.subVectors(r,i),s=(d-h)/(d-h+(p-g)),e.copy(i).addScaledVector(Fs,s);const u=1/(m+_+f);return o=_*u,s=f*u,e.copy(n).addScaledVector(Pn,o).addScaledVector(Rn,s)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}}let fl=0;class hi extends Mn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:fl++}),this.uuid=li(),this.name="",this.type="Material",this.blending=Vn,this.side=en,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.blendSrc=Sa,this.blendDst=ya,this.blendEquation=Bn,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.depthFunc=Tr,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Jo,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Zi,this.stencilZFail=Zi,this.stencilZPass=Zi,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBuild(){}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(const e in t){const n=t[e];if(n===void 0){console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);continue}const i=this[e];if(i===void 0){console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);continue}i&&i.isColor?i.set(n):i&&i.isVector3&&n&&n.isVector3?i.copy(n):this[e]=n}}toJSON(t){const e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});const n={metadata:{version:4.5,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Vn&&(n.blending=this.blending),this.side!==en&&(n.side=this.side),this.vertexColors&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=this.transparent),n.depthFunc=this.depthFunc,n.depthTest=this.depthTest,n.depthWrite=this.depthWrite,n.colorWrite=this.colorWrite,n.stencilWrite=this.stencilWrite,n.stencilWriteMask=this.stencilWriteMask,n.stencilFunc=this.stencilFunc,n.stencilRef=this.stencilRef,n.stencilFuncMask=this.stencilFuncMask,n.stencilFail=this.stencilFail,n.stencilZFail=this.stencilZFail,n.stencilZPass=this.stencilZPass,this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaToCoverage===!0&&(n.alphaToCoverage=this.alphaToCoverage),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=this.premultipliedAlpha),this.forceSinglePass===!0&&(n.forceSinglePass=this.forceSinglePass),this.wireframe===!0&&(n.wireframe=this.wireframe),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=this.flatShading),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function i(r){const o=[];for(const s in r){const c=r[s];delete c.metadata,o.push(c)}return o}if(e){const r=i(t.textures),o=i(t.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;const e=t.clippingPlanes;let n=null;if(e!==null){const i=e.length;n=new Array(i);for(let r=0;r!==i;++r)n[r]=e[r].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}}const Ua={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Le={h:0,s:0,l:0},Mi={h:0,s:0,l:0};function dr(a,t,e){return e<0&&(e+=1),e>1&&(e-=1),e<1/6?a+(t-a)*6*e:e<1/2?t:e<2/3?a+(t-a)*6*(2/3-e):a}class Ot{constructor(t,e,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,e===void 0&&n===void 0?this.set(t):this.setRGB(t,e,n)}set(t){return t&&t.isColor?this.copy(t):typeof t=="number"?this.setHex(t):typeof t=="string"&&this.setStyle(t),this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=Oe){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,_e.toWorkingColorSpace(this,e),this}setRGB(t,e,n,i=_e.workingColorSpace){return this.r=t,this.g=e,this.b=n,_e.toWorkingColorSpace(this,i),this}setHSL(t,e,n,i=_e.workingColorSpace){if(t=Zo(t,1),e=ne(e,0,1),n=ne(n,0,1),e===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+e):n+e-n*e,o=2*n-r;this.r=dr(o,r,t+1/3),this.g=dr(o,r,t),this.b=dr(o,r,t-1/3)}return _e.toWorkingColorSpace(this,i),this}setStyle(t,e=Oe){function n(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let i;if(i=/^(\w+)\(([^\)]*)\)/.exec(t)){let r;const o=i[1],s=i[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(s))return this.r=Math.min(255,parseInt(r[1],10))/255,this.g=Math.min(255,parseInt(r[2],10))/255,this.b=Math.min(255,parseInt(r[3],10))/255,_e.toWorkingColorSpace(this,e),n(r[4]),this;if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(s))return this.r=Math.min(100,parseInt(r[1],10))/100,this.g=Math.min(100,parseInt(r[2],10))/100,this.b=Math.min(100,parseInt(r[3],10))/100,_e.toWorkingColorSpace(this,e),n(r[4]),this;break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(s)){const c=parseFloat(r[1])/360,l=parseFloat(r[2])/100,h=parseFloat(r[3])/100;return n(r[4]),this.setHSL(c,l,h,e)}break;default:console.warn("THREE.Color: Unknown color model "+t)}}else if(i=/^\#([A-Fa-f\d]+)$/.exec(t)){const r=i[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(o===6)return this.setHex(parseInt(r,16),e);console.warn("THREE.Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=Oe){const n=Ua[t.toLowerCase()];return n!==void 0?this.setHex(n,e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=Hn(t.r),this.g=Hn(t.g),this.b=Hn(t.b),this}copyLinearToSRGB(t){return this.r=tr(t.r),this.g=tr(t.g),this.b=tr(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=Oe){return _e.fromWorkingColorSpace(ce.copy(this),t),ne(ce.r*255,0,255)<<16^ne(ce.g*255,0,255)<<8^ne(ce.b*255,0,255)<<0}getHexString(t=Oe){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=_e.workingColorSpace){_e.fromWorkingColorSpace(ce.copy(this),e);const n=ce.r,i=ce.g,r=ce.b,o=Math.max(n,i,r),s=Math.min(n,i,r);let c,l;const h=(s+o)/2;if(s===o)c=0,l=0;else{const d=o-s;switch(l=h<=.5?d/(o+s):d/(2-o-s),o){case n:c=(i-r)/d+(i<r?6:0);break;case i:c=(r-n)/d+2;break;case r:c=(n-i)/d+4;break}c/=6}return t.h=c,t.s=l,t.l=h,t}getRGB(t,e=_e.workingColorSpace){return _e.fromWorkingColorSpace(ce.copy(this),e),t.r=ce.r,t.g=ce.g,t.b=ce.b,t}getStyle(t=Oe){_e.fromWorkingColorSpace(ce.copy(this),t);const e=ce.r,n=ce.g,i=ce.b;return t!==Oe?`color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${i.toFixed(3)})`:`rgb(${e*255|0},${n*255|0},${i*255|0})`}offsetHSL(t,e,n){return this.getHSL(Le),Le.h+=t,Le.s+=e,Le.l+=n,this.setHSL(Le.h,Le.s,Le.l),this}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(Le),t.getHSL(Mi);const n=$i(Le.h,Mi.h,e),i=$i(Le.s,Mi.s,e),r=$i(Le.l,Mi.l,e);return this.setHSL(n,i,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){const e=this.r,n=this.g,i=this.b,r=t.elements;return this.r=r[0]*e+r[3]*n+r[6]*i,this.g=r[1]*e+r[4]*n+r[7]*i,this.b=r[2]*e+r[5]*n+r[8]*i,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const ce=new Ot;Ot.NAMES=Ua;class Na extends hi{constructor(t){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Ot(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.combine=ba,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}}const Zt=new P,Si=new dt;class Ie{constructor(t,e,n=!1){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=n,this.usage=Ts,this.updateRange={offset:0,count:-1},this.version=0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let i=0,r=this.itemSize;i<r;i++)this.array[t+i]=e.array[n+i];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,n=this.count;e<n;e++)Si.fromBufferAttribute(this,e),Si.applyMatrix3(t),this.setXY(e,Si.x,Si.y);else if(this.itemSize===3)for(let e=0,n=this.count;e<n;e++)Zt.fromBufferAttribute(this,e),Zt.applyMatrix3(t),this.setXYZ(e,Zt.x,Zt.y,Zt.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)Zt.fromBufferAttribute(this,e),Zt.applyMatrix4(t),this.setXYZ(e,Zt.x,Zt.y,Zt.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)Zt.fromBufferAttribute(this,e),Zt.applyNormalMatrix(t),this.setXYZ(e,Zt.x,Zt.y,Zt.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)Zt.fromBufferAttribute(this,e),Zt.transformDirection(t),this.setXYZ(e,Zt.x,Zt.y,Zt.z);return this}set(t,e=0){return this.array.set(t,e),this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=di(e,this.array)),e}setX(t,e){return this.normalized&&(e=ge(e,this.array)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=di(e,this.array)),e}setY(t,e){return this.normalized&&(e=ge(e,this.array)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=di(e,this.array)),e}setZ(t,e){return this.normalized&&(e=ge(e,this.array)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=di(e,this.array)),e}setW(t,e){return this.normalized&&(e=ge(e,this.array)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.normalized&&(e=ge(e,this.array),n=ge(n,this.array)),this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,i){return t*=this.itemSize,this.normalized&&(e=ge(e,this.array),n=ge(n,this.array),i=ge(i,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this}setXYZW(t,e,n,i,r){return t*=this.itemSize,this.normalized&&(e=ge(e,this.array),n=ge(n,this.array),i=ge(i,this.array),r=ge(r,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(t.name=this.name),this.usage!==Ts&&(t.usage=this.usage),(this.updateRange.offset!==0||this.updateRange.count!==-1)&&(t.updateRange=this.updateRange),t}copyColorsArray(){console.error("THREE.BufferAttribute: copyColorsArray() was removed in r144.")}copyVector2sArray(){console.error("THREE.BufferAttribute: copyVector2sArray() was removed in r144.")}copyVector3sArray(){console.error("THREE.BufferAttribute: copyVector3sArray() was removed in r144.")}copyVector4sArray(){console.error("THREE.BufferAttribute: copyVector4sArray() was removed in r144.")}}class Oa extends Ie{constructor(t,e,n){super(new Uint16Array(t),e,n)}}class Fa extends Ie{constructor(t,e,n){super(new Uint32Array(t),e,n)}}class se extends Ie{constructor(t,e,n){super(new Float32Array(t),e,n)}}let pl=0;const ye=new Kt,fr=new re,Dn=new P,xe=new ci,ti=new ci,ee=new P;class Ee extends Mn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:pl++}),this.uuid=li(),this.name="",this.type="BufferGeometry",this.index=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(La(t)?Fa:Oa)(t,1):this.index=t,this}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return this.attributes[t]!==void 0}addGroup(t,e,n=0){this.groups.push({start:t,count:e,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){const e=this.attributes.position;e!==void 0&&(e.applyMatrix4(t),e.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new It().getNormalMatrix(t);n.applyNormalMatrix(r),n.needsUpdate=!0}const i=this.attributes.tangent;return i!==void 0&&(i.transformDirection(t),i.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(t){return ye.makeRotationFromQuaternion(t),this.applyMatrix4(ye),this}rotateX(t){return ye.makeRotationX(t),this.applyMatrix4(ye),this}rotateY(t){return ye.makeRotationY(t),this.applyMatrix4(ye),this}rotateZ(t){return ye.makeRotationZ(t),this.applyMatrix4(ye),this}translate(t,e,n){return ye.makeTranslation(t,e,n),this.applyMatrix4(ye),this}scale(t,e,n){return ye.makeScale(t,e,n),this.applyMatrix4(ye),this}lookAt(t){return fr.lookAt(t),fr.updateMatrix(),this.applyMatrix4(fr.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Dn).negate(),this.translate(Dn.x,Dn.y,Dn.z),this}setFromPoints(t){const e=[];for(let n=0,i=t.length;n<i;n++){const r=t[n];e.push(r.x,r.y,r.z||0)}return this.setAttribute("position",new se(e,3)),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new ci);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error('THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box. Alternatively set "mesh.frustumCulled" to "false".',this),this.boundingBox.set(new P(-1/0,-1/0,-1/0),new P(1/0,1/0,1/0));return}if(t!==void 0){if(this.boundingBox.setFromBufferAttribute(t),e)for(let n=0,i=e.length;n<i;n++){const r=e[n];xe.setFromBufferAttribute(r),this.morphTargetsRelative?(ee.addVectors(this.boundingBox.min,xe.min),this.boundingBox.expandByPoint(ee),ee.addVectors(this.boundingBox.max,xe.max),this.boundingBox.expandByPoint(ee)):(this.boundingBox.expandByPoint(xe.min),this.boundingBox.expandByPoint(xe.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Nr);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error('THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere. Alternatively set "mesh.frustumCulled" to "false".',this),this.boundingSphere.set(new P,1/0);return}if(t){const n=this.boundingSphere.center;if(xe.setFromBufferAttribute(t),e)for(let r=0,o=e.length;r<o;r++){const s=e[r];ti.setFromBufferAttribute(s),this.morphTargetsRelative?(ee.addVectors(xe.min,ti.min),xe.expandByPoint(ee),ee.addVectors(xe.max,ti.max),xe.expandByPoint(ee)):(xe.expandByPoint(ti.min),xe.expandByPoint(ti.max))}xe.getCenter(n);let i=0;for(let r=0,o=t.count;r<o;r++)ee.fromBufferAttribute(t,r),i=Math.max(i,n.distanceToSquared(ee));if(e)for(let r=0,o=e.length;r<o;r++){const s=e[r],c=this.morphTargetsRelative;for(let l=0,h=s.count;l<h;l++)ee.fromBufferAttribute(s,l),c&&(Dn.fromBufferAttribute(t,l),ee.add(Dn)),i=Math.max(i,n.distanceToSquared(ee))}this.boundingSphere.radius=Math.sqrt(i),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const t=this.index,e=this.attributes;if(t===null||e.position===void 0||e.normal===void 0||e.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.array,i=e.position.array,r=e.normal.array,o=e.uv.array,s=i.length/3;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Ie(new Float32Array(4*s),4));const c=this.getAttribute("tangent").array,l=[],h=[];for(let T=0;T<s;T++)l[T]=new P,h[T]=new P;const d=new P,f=new P,p=new P,g=new dt,_=new dt,m=new dt,u=new P,E=new P;function x(T,X,q){d.fromArray(i,T*3),f.fromArray(i,X*3),p.fromArray(i,q*3),g.fromArray(o,T*2),_.fromArray(o,X*2),m.fromArray(o,q*2),f.sub(d),p.sub(d),_.sub(g),m.sub(g);const I=1/(_.x*m.y-m.x*_.y);isFinite(I)&&(u.copy(f).multiplyScalar(m.y).addScaledVector(p,-_.y).multiplyScalar(I),E.copy(p).multiplyScalar(_.x).addScaledVector(f,-m.x).multiplyScalar(I),l[T].add(u),l[X].add(u),l[q].add(u),h[T].add(E),h[X].add(E),h[q].add(E))}let b=this.groups;b.length===0&&(b=[{start:0,count:n.length}]);for(let T=0,X=b.length;T<X;++T){const q=b[T],I=q.start,N=q.count;for(let B=I,Q=I+N;B<Q;B+=3)x(n[B+0],n[B+1],n[B+2])}const y=new P,C=new P,D=new P,R=new P;function M(T){D.fromArray(r,T*3),R.copy(D);const X=l[T];y.copy(X),y.sub(D.multiplyScalar(D.dot(X))).normalize(),C.crossVectors(R,X);const I=C.dot(h[T])<0?-1:1;c[T*4]=y.x,c[T*4+1]=y.y,c[T*4+2]=y.z,c[T*4+3]=I}for(let T=0,X=b.length;T<X;++T){const q=b[T],I=q.start,N=q.count;for(let B=I,Q=I+N;B<Q;B+=3)M(n[B+0]),M(n[B+1]),M(n[B+2])}}computeVertexNormals(){const t=this.index,e=this.getAttribute("position");if(e!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new Ie(new Float32Array(e.count*3),3),this.setAttribute("normal",n);else for(let f=0,p=n.count;f<p;f++)n.setXYZ(f,0,0,0);const i=new P,r=new P,o=new P,s=new P,c=new P,l=new P,h=new P,d=new P;if(t)for(let f=0,p=t.count;f<p;f+=3){const g=t.getX(f+0),_=t.getX(f+1),m=t.getX(f+2);i.fromBufferAttribute(e,g),r.fromBufferAttribute(e,_),o.fromBufferAttribute(e,m),h.subVectors(o,r),d.subVectors(i,r),h.cross(d),s.fromBufferAttribute(n,g),c.fromBufferAttribute(n,_),l.fromBufferAttribute(n,m),s.add(h),c.add(h),l.add(h),n.setXYZ(g,s.x,s.y,s.z),n.setXYZ(_,c.x,c.y,c.z),n.setXYZ(m,l.x,l.y,l.z)}else for(let f=0,p=e.count;f<p;f+=3)i.fromBufferAttribute(e,f+0),r.fromBufferAttribute(e,f+1),o.fromBufferAttribute(e,f+2),h.subVectors(o,r),d.subVectors(i,r),h.cross(d),n.setXYZ(f+0,h.x,h.y,h.z),n.setXYZ(f+1,h.x,h.y,h.z),n.setXYZ(f+2,h.x,h.y,h.z);this.normalizeNormals(),n.needsUpdate=!0}}merge(){return console.error("THREE.BufferGeometry.merge() has been removed. Use THREE.BufferGeometryUtils.mergeGeometries() instead."),this}normalizeNormals(){const t=this.attributes.normal;for(let e=0,n=t.count;e<n;e++)ee.fromBufferAttribute(t,e),ee.normalize(),t.setXYZ(e,ee.x,ee.y,ee.z)}toNonIndexed(){function t(s,c){const l=s.array,h=s.itemSize,d=s.normalized,f=new l.constructor(c.length*h);let p=0,g=0;for(let _=0,m=c.length;_<m;_++){s.isInterleavedBufferAttribute?p=c[_]*s.data.stride+s.offset:p=c[_]*h;for(let u=0;u<h;u++)f[g++]=l[p++]}return new Ie(f,h,d)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const e=new Ee,n=this.index.array,i=this.attributes;for(const s in i){const c=i[s],l=t(c,n);e.setAttribute(s,l)}const r=this.morphAttributes;for(const s in r){const c=[],l=r[s];for(let h=0,d=l.length;h<d;h++){const f=l[h],p=t(f,n);c.push(p)}e.morphAttributes[s]=c}e.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let s=0,c=o.length;s<c;s++){const l=o[s];e.addGroup(l.start,l.count,l.materialIndex)}return e}toJSON(){const t={metadata:{version:4.5,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.type,this.name!==""&&(t.name=this.name),Object.keys(this.userData).length>0&&(t.userData=this.userData),this.parameters!==void 0){const c=this.parameters;for(const l in c)c[l]!==void 0&&(t[l]=c[l]);return t}t.data={attributes:{}};const e=this.index;e!==null&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});const n=this.attributes;for(const c in n){const l=n[c];t.data.attributes[c]=l.toJSON(t.data)}const i={};let r=!1;for(const c in this.morphAttributes){const l=this.morphAttributes[c],h=[];for(let d=0,f=l.length;d<f;d++){const p=l[d];h.push(p.toJSON(t.data))}h.length>0&&(i[c]=h,r=!0)}r&&(t.data.morphAttributes=i,t.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(t.data.groups=JSON.parse(JSON.stringify(o)));const s=this.boundingSphere;return s!==null&&(t.data.boundingSphere={center:s.center.toArray(),radius:s.radius}),t}clone(){return new this.constructor().copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const e={};this.name=t.name;const n=t.index;n!==null&&this.setIndex(n.clone(e));const i=t.attributes;for(const l in i){const h=i[l];this.setAttribute(l,h.clone(e))}const r=t.morphAttributes;for(const l in r){const h=[],d=r[l];for(let f=0,p=d.length;f<p;f++)h.push(d[f].clone(e));this.morphAttributes[l]=h}this.morphTargetsRelative=t.morphTargetsRelative;const o=t.groups;for(let l=0,h=o.length;l<h;l++){const d=o[l];this.addGroup(d.start,d.count,d.materialIndex)}const s=t.boundingBox;s!==null&&(this.boundingBox=s.clone());const c=t.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const zs=new Kt,Ne=new Ia,yi=new Nr,Bs=new P,In=new P,Un=new P,Nn=new P,pr=new P,bi=new P,wi=new dt,Ei=new dt,Ti=new dt,Gs=new P,Vs=new P,ks=new P,Ai=new P,Ci=new P;class ze extends re{constructor(t=new Ee,e=new Na){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),t.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=t.material,this.geometry=t.geometry,this}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const i=e[n[0]];if(i!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=i.length;r<o;r++){const s=i[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[s]=r}}}}getVertexPosition(t,e){const n=this.geometry,i=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;e.fromBufferAttribute(i,t);const s=this.morphTargetInfluences;if(r&&s){bi.set(0,0,0);for(let c=0,l=r.length;c<l;c++){const h=s[c],d=r[c];h!==0&&(pr.fromBufferAttribute(d,t),o?bi.addScaledVector(pr,h):bi.addScaledVector(pr.sub(e),h))}e.add(bi)}return this.isSkinnedMesh&&this.applyBoneTransform(t,e),e}raycast(t,e){const n=this.geometry,i=this.material,r=this.matrixWorld;if(i===void 0||(n.boundingSphere===null&&n.computeBoundingSphere(),yi.copy(n.boundingSphere),yi.applyMatrix4(r),Ne.copy(t.ray).recast(t.near),yi.containsPoint(Ne.origin)===!1&&(Ne.intersectSphere(yi,Bs)===null||Ne.origin.distanceToSquared(Bs)>(t.far-t.near)**2))||(zs.copy(r).invert(),Ne.copy(t.ray).applyMatrix4(zs),n.boundingBox!==null&&Ne.intersectsBox(n.boundingBox)===!1))return;let o;const s=n.index,c=n.attributes.position,l=n.attributes.uv,h=n.attributes.uv2,d=n.attributes.normal,f=n.groups,p=n.drawRange;if(s!==null)if(Array.isArray(i))for(let g=0,_=f.length;g<_;g++){const m=f[g],u=i[m.materialIndex],E=Math.max(m.start,p.start),x=Math.min(s.count,Math.min(m.start+m.count,p.start+p.count));for(let b=E,y=x;b<y;b+=3){const C=s.getX(b),D=s.getX(b+1),R=s.getX(b+2);o=Li(this,u,t,Ne,l,h,d,C,D,R),o&&(o.faceIndex=Math.floor(b/3),o.face.materialIndex=m.materialIndex,e.push(o))}}else{const g=Math.max(0,p.start),_=Math.min(s.count,p.start+p.count);for(let m=g,u=_;m<u;m+=3){const E=s.getX(m),x=s.getX(m+1),b=s.getX(m+2);o=Li(this,i,t,Ne,l,h,d,E,x,b),o&&(o.faceIndex=Math.floor(m/3),e.push(o))}}else if(c!==void 0)if(Array.isArray(i))for(let g=0,_=f.length;g<_;g++){const m=f[g],u=i[m.materialIndex],E=Math.max(m.start,p.start),x=Math.min(c.count,Math.min(m.start+m.count,p.start+p.count));for(let b=E,y=x;b<y;b+=3){const C=b,D=b+1,R=b+2;o=Li(this,u,t,Ne,l,h,d,C,D,R),o&&(o.faceIndex=Math.floor(b/3),o.face.materialIndex=m.materialIndex,e.push(o))}}else{const g=Math.max(0,p.start),_=Math.min(c.count,p.start+p.count);for(let m=g,u=_;m<u;m+=3){const E=m,x=m+1,b=m+2;o=Li(this,i,t,Ne,l,h,d,E,x,b),o&&(o.faceIndex=Math.floor(m/3),e.push(o))}}}}function ml(a,t,e,n,i,r,o,s){let c;if(t.side===pe?c=n.intersectTriangle(o,r,i,!0,s):c=n.intersectTriangle(i,r,o,t.side===en,s),c===null)return null;Ci.copy(s),Ci.applyMatrix4(a.matrixWorld);const l=e.ray.origin.distanceTo(Ci);return l<e.near||l>e.far?null:{distance:l,point:Ci.clone(),object:a}}function Li(a,t,e,n,i,r,o,s,c,l){a.getVertexPosition(s,In),a.getVertexPosition(c,Un),a.getVertexPosition(l,Nn);const h=ml(a,t,e,n,In,Un,Nn,Ai);if(h){i&&(wi.fromBufferAttribute(i,s),Ei.fromBufferAttribute(i,c),Ti.fromBufferAttribute(i,l),h.uv=Pe.getInterpolation(Ai,In,Un,Nn,wi,Ei,Ti,new dt)),r&&(wi.fromBufferAttribute(r,s),Ei.fromBufferAttribute(r,c),Ti.fromBufferAttribute(r,l),h.uv2=Pe.getInterpolation(Ai,In,Un,Nn,wi,Ei,Ti,new dt)),o&&(Gs.fromBufferAttribute(o,s),Vs.fromBufferAttribute(o,c),ks.fromBufferAttribute(o,l),h.normal=Pe.getInterpolation(Ai,In,Un,Nn,Gs,Vs,ks,new P),h.normal.dot(n.direction)>0&&h.normal.multiplyScalar(-1));const d={a:s,b:c,c:l,normal:new P,materialIndex:0};Pe.getNormal(In,Un,Nn,d.normal),h.face=d}return h}class jn extends Ee{constructor(t=1,e=1,n=1,i=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:n,widthSegments:i,heightSegments:r,depthSegments:o};const s=this;i=Math.floor(i),r=Math.floor(r),o=Math.floor(o);const c=[],l=[],h=[],d=[];let f=0,p=0;g("z","y","x",-1,-1,n,e,t,o,r,0),g("z","y","x",1,-1,n,e,-t,o,r,1),g("x","z","y",1,1,t,n,e,i,o,2),g("x","z","y",1,-1,t,n,-e,i,o,3),g("x","y","z",1,-1,t,e,n,i,r,4),g("x","y","z",-1,-1,t,e,-n,i,r,5),this.setIndex(c),this.setAttribute("position",new se(l,3)),this.setAttribute("normal",new se(h,3)),this.setAttribute("uv",new se(d,2));function g(_,m,u,E,x,b,y,C,D,R,M){const T=b/D,X=y/R,q=b/2,I=y/2,N=C/2,B=D+1,Q=R+1;let et=0,j=0;const Z=new P;for(let nt=0;nt<Q;nt++){const vt=nt*X-I;for(let ct=0;ct<B;ct++){const V=ct*T-q;Z[_]=V*E,Z[m]=vt*x,Z[u]=N,l.push(Z.x,Z.y,Z.z),Z[_]=0,Z[m]=0,Z[u]=C>0?1:-1,h.push(Z.x,Z.y,Z.z),d.push(ct/D),d.push(1-nt/R),et+=1}}for(let nt=0;nt<R;nt++)for(let vt=0;vt<D;vt++){const ct=f+vt+B*nt,V=f+vt+B*(nt+1),J=f+(vt+1)+B*(nt+1),st=f+(vt+1)+B*nt;c.push(ct,V,st),c.push(V,J,st),j+=6}s.addGroup(p,j,M),p+=j,f+=et}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new jn(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}}function Yn(a){const t={};for(const e in a){t[e]={};for(const n in a[e]){const i=a[e][n];i&&(i.isColor||i.isMatrix3||i.isMatrix4||i.isVector2||i.isVector3||i.isVector4||i.isTexture||i.isQuaternion)?i.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),t[e][n]=null):t[e][n]=i.clone():Array.isArray(i)?t[e][n]=i.slice():t[e][n]=i}}return t}function de(a){const t={};for(let e=0;e<a.length;e++){const n=Yn(a[e]);for(const i in n)t[i]=n[i]}return t}function gl(a){const t=[];for(let e=0;e<a.length;e++)t.push(a[e].clone());return t}function za(a){return a.getRenderTarget()===null&&a.outputEncoding===Gt?Oe:oi}const _l={clone:Yn,merge:de};var vl=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,xl=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class xn extends hi{constructor(t){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=vl,this.fragmentShader=xl,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={derivatives:!1,fragDepth:!1,drawBuffers:!1,shaderTextureLOD:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv2:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=Yn(t.uniforms),this.uniformsGroups=gl(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){const e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(const i in this.uniforms){const o=this.uniforms[i].value;o&&o.isTexture?e.uniforms[i]={type:"t",value:o.toJSON(t).uuid}:o&&o.isColor?e.uniforms[i]={type:"c",value:o.getHex()}:o&&o.isVector2?e.uniforms[i]={type:"v2",value:o.toArray()}:o&&o.isVector3?e.uniforms[i]={type:"v3",value:o.toArray()}:o&&o.isVector4?e.uniforms[i]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?e.uniforms[i]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?e.uniforms[i]={type:"m4",value:o.toArray()}:e.uniforms[i]={value:o}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader;const n={};for(const i in this.extensions)this.extensions[i]===!0&&(n[i]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}}class Ba extends re{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Kt,this.projectionMatrix=new Kt,this.projectionMatrixInverse=new Kt}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(-e[8],-e[9],-e[10]).normalize()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}class we extends Ba{constructor(t=50,e=1,n=.1,i=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=i,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){const e=.5*this.getFilmHeight()/t;this.fov=Dr*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){const t=Math.tan(Ki*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return Dr*2*Math.atan(Math.tan(Ki*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}setViewOffset(t,e,n,i,r,o){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=this.near;let e=t*Math.tan(Ki*.5*this.fov)/this.zoom,n=2*e,i=this.aspect*n,r=-.5*i;const o=this.view;if(this.view!==null&&this.view.enabled){const c=o.fullWidth,l=o.fullHeight;r+=o.offsetX*i/c,e-=o.offsetY*n/l,i*=o.width/c,n*=o.height/l}const s=this.filmOffset;s!==0&&(r+=t*s/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+i,e,e-n,t,this.far),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}}const On=-90,Fn=1;class Ml extends re{constructor(t,e,n){super(),this.type="CubeCamera",this.renderTarget=n;const i=new we(On,Fn,t,e);i.layers=this.layers,i.up.set(0,1,0),i.lookAt(1,0,0),this.add(i);const r=new we(On,Fn,t,e);r.layers=this.layers,r.up.set(0,1,0),r.lookAt(-1,0,0),this.add(r);const o=new we(On,Fn,t,e);o.layers=this.layers,o.up.set(0,0,-1),o.lookAt(0,1,0),this.add(o);const s=new we(On,Fn,t,e);s.layers=this.layers,s.up.set(0,0,1),s.lookAt(0,-1,0),this.add(s);const c=new we(On,Fn,t,e);c.layers=this.layers,c.up.set(0,1,0),c.lookAt(0,0,1),this.add(c);const l=new we(On,Fn,t,e);l.layers=this.layers,l.up.set(0,1,0),l.lookAt(0,0,-1),this.add(l)}update(t,e){this.parent===null&&this.updateMatrixWorld();const n=this.renderTarget,[i,r,o,s,c,l]=this.children,h=t.getRenderTarget(),d=t.toneMapping,f=t.xr.enabled;t.toneMapping=je,t.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,t.setRenderTarget(n,0),t.render(e,i),t.setRenderTarget(n,1),t.render(e,r),t.setRenderTarget(n,2),t.render(e,o),t.setRenderTarget(n,3),t.render(e,s),t.setRenderTarget(n,4),t.render(e,c),n.texture.generateMipmaps=p,t.setRenderTarget(n,5),t.render(e,l),t.setRenderTarget(h),t.toneMapping=d,t.xr.enabled=f,n.texture.needsPMREMUpdate=!0}}class Ga extends Me{constructor(t,e,n,i,r,o,s,c,l,h){t=t!==void 0?t:[],e=e!==void 0?e:Wn,super(t,e,n,i,r,o,s,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}}class Sl extends _n{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;const n={width:t,height:t,depth:1},i=[n,n,n,n,n,n];this.texture=new Ga(i,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.encoding),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=e.generateMipmaps!==void 0?e.generateMipmaps:!1,this.texture.minFilter=e.minFilter!==void 0?e.minFilter:be}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.encoding=e.encoding,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},i=new jn(5,5,5),r=new xn({name:"CubemapFromEquirect",uniforms:Yn(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:pe,blending:tn});r.uniforms.tEquirect.value=e;const o=new ze(i,r),s=e.minFilter;return e.minFilter===si&&(e.minFilter=be),new Ml(1,10,this).update(t,o),e.minFilter=s,o.geometry.dispose(),o.material.dispose(),this}clear(t,e,n,i){const r=t.getRenderTarget();for(let o=0;o<6;o++)t.setRenderTarget(this,o),t.clear(e,n,i);t.setRenderTarget(r)}}const mr=new P,yl=new P,bl=new It;class cn{constructor(t=new P(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,i){return this.normal.set(t,e,n),this.constant=i,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){const i=mr.subVectors(n,e).cross(yl.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(i,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){const t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e){const n=t.delta(mr),i=this.normal.dot(n);if(i===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;const r=-(t.start.dot(this.normal)+this.constant)/i;return r<0||r>1?null:e.copy(t.start).addScaledVector(n,r)}intersectsLine(t){const e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){const n=e||bl.getNormalMatrix(t),i=this.coplanarPoint(mr).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-i.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}}const ln=new Nr,Pi=new P;class Fr{constructor(t=new cn,e=new cn,n=new cn,i=new cn,r=new cn,o=new cn){this.planes=[t,e,n,i,r,o]}set(t,e,n,i,r,o){const s=this.planes;return s[0].copy(t),s[1].copy(e),s[2].copy(n),s[3].copy(i),s[4].copy(r),s[5].copy(o),this}copy(t){const e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t){const e=this.planes,n=t.elements,i=n[0],r=n[1],o=n[2],s=n[3],c=n[4],l=n[5],h=n[6],d=n[7],f=n[8],p=n[9],g=n[10],_=n[11],m=n[12],u=n[13],E=n[14],x=n[15];return e[0].setComponents(s-i,d-c,_-f,x-m).normalize(),e[1].setComponents(s+i,d+c,_+f,x+m).normalize(),e[2].setComponents(s+r,d+l,_+p,x+u).normalize(),e[3].setComponents(s-r,d-l,_-p,x-u).normalize(),e[4].setComponents(s-o,d-h,_-g,x-E).normalize(),e[5].setComponents(s+o,d+h,_+g,x+E).normalize(),this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),ln.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{const e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),ln.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(ln)}intersectsSprite(t){return ln.center.set(0,0,0),ln.radius=.7071067811865476,ln.applyMatrix4(t.matrixWorld),this.intersectsSphere(ln)}intersectsSphere(t){const e=this.planes,n=t.center,i=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(n)<i)return!1;return!0}intersectsBox(t){const e=this.planes;for(let n=0;n<6;n++){const i=e[n];if(Pi.x=i.normal.x>0?t.max.x:t.min.x,Pi.y=i.normal.y>0?t.max.y:t.min.y,Pi.z=i.normal.z>0?t.max.z:t.min.z,i.distanceToPoint(Pi)<0)return!1}return!0}containsPoint(t){const e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}function Va(){let a=null,t=!1,e=null,n=null;function i(r,o){e(r,o),n=a.requestAnimationFrame(i)}return{start:function(){t!==!0&&e!==null&&(n=a.requestAnimationFrame(i),t=!0)},stop:function(){a.cancelAnimationFrame(n),t=!1},setAnimationLoop:function(r){e=r},setContext:function(r){a=r}}}function wl(a,t){const e=t.isWebGL2,n=new WeakMap;function i(l,h){const d=l.array,f=l.usage,p=a.createBuffer();a.bindBuffer(h,p),a.bufferData(h,d,f),l.onUploadCallback();let g;if(d instanceof Float32Array)g=5126;else if(d instanceof Uint16Array)if(l.isFloat16BufferAttribute)if(e)g=5131;else throw new Error("THREE.WebGLAttributes: Usage of Float16BufferAttribute requires WebGL2.");else g=5123;else if(d instanceof Int16Array)g=5122;else if(d instanceof Uint32Array)g=5125;else if(d instanceof Int32Array)g=5124;else if(d instanceof Int8Array)g=5120;else if(d instanceof Uint8Array)g=5121;else if(d instanceof Uint8ClampedArray)g=5121;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+d);return{buffer:p,type:g,bytesPerElement:d.BYTES_PER_ELEMENT,version:l.version}}function r(l,h,d){const f=h.array,p=h.updateRange;a.bindBuffer(d,l),p.count===-1?a.bufferSubData(d,0,f):(e?a.bufferSubData(d,p.offset*f.BYTES_PER_ELEMENT,f,p.offset,p.count):a.bufferSubData(d,p.offset*f.BYTES_PER_ELEMENT,f.subarray(p.offset,p.offset+p.count)),p.count=-1),h.onUploadCallback()}function o(l){return l.isInterleavedBufferAttribute&&(l=l.data),n.get(l)}function s(l){l.isInterleavedBufferAttribute&&(l=l.data);const h=n.get(l);h&&(a.deleteBuffer(h.buffer),n.delete(l))}function c(l,h){if(l.isGLBufferAttribute){const f=n.get(l);(!f||f.version<l.version)&&n.set(l,{buffer:l.buffer,type:l.type,bytesPerElement:l.elementSize,version:l.version});return}l.isInterleavedBufferAttribute&&(l=l.data);const d=n.get(l);d===void 0?n.set(l,i(l,h)):d.version<l.version&&(r(d.buffer,l,h),d.version=l.version)}return{get:o,remove:s,update:c}}class zi extends Ee{constructor(t=1,e=1,n=1,i=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:n,heightSegments:i};const r=t/2,o=e/2,s=Math.floor(n),c=Math.floor(i),l=s+1,h=c+1,d=t/s,f=e/c,p=[],g=[],_=[],m=[];for(let u=0;u<h;u++){const E=u*f-o;for(let x=0;x<l;x++){const b=x*d-r;g.push(b,-E,0),_.push(0,0,1),m.push(x/s),m.push(1-u/c)}}for(let u=0;u<c;u++)for(let E=0;E<s;E++){const x=E+l*u,b=E+l*(u+1),y=E+1+l*(u+1),C=E+1+l*u;p.push(x,b,C),p.push(b,y,C)}this.setIndex(p),this.setAttribute("position",new se(g,3)),this.setAttribute("normal",new se(_,3)),this.setAttribute("uv",new se(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new zi(t.width,t.height,t.widthSegments,t.heightSegments)}}var El=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Tl=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Al=`#ifdef USE_ALPHATEST
	if ( diffuseColor.a < alphaTest ) discard;
#endif`,Cl=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Ll=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Pl=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Rl="vec3 transformed = vec3( position );",Dl=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Il=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Ul=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			 return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float R21 = R12;
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Nl=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = dFdx( surf_pos.xyz );
		vec3 vSigmaY = dFdy( surf_pos.xyz );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Ol=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#pragma unroll_loop_start
	for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
		plane = clippingPlanes[ i ];
		if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
	}
	#pragma unroll_loop_end
	#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
		bool clipped = true;
		#pragma unroll_loop_start
		for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
		}
		#pragma unroll_loop_end
		if ( clipped ) discard;
	#endif
#endif`,Fl=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,zl=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Bl=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Gl=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Vl=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,kl=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )
	varying vec3 vColor;
#endif`,Hl=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif`,Wl=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
struct GeometricContext {
	vec3 position;
	vec3 normal;
	vec3 viewDir;
#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal;
#endif
};
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
float luminance( const in vec3 rgb ) {
	const vec3 weights = vec3( 0.2126729, 0.7151522, 0.0721750 );
	return dot( weights, rgb );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,ql=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_v0 0.339
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_v1 0.276
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_v4 0.046
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_v5 0.016
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_v6 0.0038
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Xl=`vec3 transformedNormal = objectNormal;
#ifdef USE_INSTANCING
	mat3 m = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( m[ 0 ], m[ 0 ] ), dot( m[ 1 ], m[ 1 ] ), dot( m[ 2 ], m[ 2 ] ) );
	transformedNormal = m * transformedNormal;
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	vec3 transformedTangent = ( modelViewMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Yl=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,jl=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Jl=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Zl=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Kl="gl_FragColor = linearToOutputTexel( gl_FragColor );",$l=`vec4 LinearToLinear( in vec4 value ) {
	return value;
}
vec4 LinearTosRGB( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Ql=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,tc=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,ec=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,nc=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,ic=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,rc=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,sc=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,ac=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,oc=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,lc=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,cc=`#ifdef USE_LIGHTMAP
	vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
	vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
	reflectedLight.indirectDiffuse += lightMapIrradiance;
#endif`,hc=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,uc=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,dc=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in GeometricContext geometry, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in GeometricContext geometry, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,fc=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	#if defined ( LEGACY_LIGHTS )
		if ( cutoffDistance > 0.0 && decayExponent > 0.0 ) {
			return pow( saturate( - lightDistance / cutoffDistance + 1.0 ), decayExponent );
		}
		return 1.0;
	#else
		float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
		if ( cutoffDistance > 0.0 ) {
			distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
		}
		return distanceFalloff;
	#endif
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, const in GeometricContext geometry, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometry.position;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in GeometricContext geometry, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometry.position;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,pc=`#if defined( USE_ENVMAP )
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#if defined( ENVMAP_TYPE_CUBE_UV )
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#if defined( ENVMAP_TYPE_CUBE_UV )
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
#endif`,mc=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,gc=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in GeometricContext geometry, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometry.normal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in GeometricContext geometry, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,_c=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,vc=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,xc=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( geometryNormal ) ), abs( dFdy( geometryNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif`,Mc=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
};
vec3 clearcoatSpecular = vec3( 0.0 );
vec3 sheenSpecular = vec3( 0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
	float D = D_GGX( alpha, dotNH );
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometry.normal;
		vec3 viewDir = geometry.viewDir;
		vec3 position = geometry.position;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometry.clearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecular += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometry.viewDir, geometry.clearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecular += irradiance * BRDF_Sheen( directLight.direction, geometry.viewDir, geometry.normal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecular += clearcoatRadiance * EnvironmentBRDF( geometry.clearcoatNormal, geometry.viewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecular += irradiance * material.sheenColor * IBLSheenBRDF( geometry.normal, geometry.viewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Sc=`
GeometricContext geometry;
geometry.position = - vViewPosition;
geometry.normal = normal;
geometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
#ifdef USE_CLEARCOAT
	geometry.clearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometry.viewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometry, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometry, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometry, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometry, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, geometry, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometry, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometry, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	irradiance += getLightProbeIrradiance( lightProbe, geometry.normal );
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,yc=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometry.normal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	radiance += getIBLRadiance( geometry.viewDir, geometry.normal, material.roughness );
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometry.viewDir, geometry.clearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,bc=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight );
#endif`,wc=`#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
	gl_FragDepthEXT = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Ec=`#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Tc=`#ifdef USE_LOGDEPTHBUF
	#ifdef USE_LOGDEPTHBUF_EXT
		varying float vFragDepth;
		varying float vIsPerspective;
	#else
		uniform float logDepthBufFC;
	#endif
#endif`,Ac=`#ifdef USE_LOGDEPTHBUF
	#ifdef USE_LOGDEPTHBUF_EXT
		vFragDepth = 1.0 + gl_Position.w;
		vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
	#else
		if ( isPerspectiveMatrix( projectionMatrix ) ) {
			gl_Position.z = log2( max( EPSILON, gl_Position.w + 1.0 ) ) * logDepthBufFC - 1.0;
			gl_Position.z *= gl_Position.w;
		}
	#endif
#endif`,Cc=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = vec4( mix( pow( sampledDiffuseColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), sampledDiffuseColor.rgb * 0.0773993808, vec3( lessThanEqual( sampledDiffuseColor.rgb, vec3( 0.04045 ) ) ) ), sampledDiffuseColor.w );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Lc=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Pc=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Rc=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Dc=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Ic=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Uc=`#if defined( USE_MORPHCOLORS ) && defined( MORPHTARGETS_TEXTURE )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Nc=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
			if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
		}
	#else
		objectNormal += morphNormal0 * morphTargetInfluences[ 0 ];
		objectNormal += morphNormal1 * morphTargetInfluences[ 1 ];
		objectNormal += morphNormal2 * morphTargetInfluences[ 2 ];
		objectNormal += morphNormal3 * morphTargetInfluences[ 3 ];
	#endif
#endif`,Oc=`#ifdef USE_MORPHTARGETS
	uniform float morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
		uniform sampler2DArray morphTargetsTexture;
		uniform ivec2 morphTargetsTextureSize;
		vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
			int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
			int y = texelIndex / morphTargetsTextureSize.x;
			int x = texelIndex - y * morphTargetsTextureSize.x;
			ivec3 morphUV = ivec3( x, y, morphTargetIndex );
			return texelFetch( morphTargetsTexture, morphUV, 0 );
		}
	#else
		#ifndef USE_MORPHNORMALS
			uniform float morphTargetInfluences[ 8 ];
		#else
			uniform float morphTargetInfluences[ 4 ];
		#endif
	#endif
#endif`,Fc=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
			if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
		}
	#else
		transformed += morphTarget0 * morphTargetInfluences[ 0 ];
		transformed += morphTarget1 * morphTargetInfluences[ 1 ];
		transformed += morphTarget2 * morphTargetInfluences[ 2 ];
		transformed += morphTarget3 * morphTargetInfluences[ 3 ];
		#ifndef USE_MORPHNORMALS
			transformed += morphTarget4 * morphTargetInfluences[ 4 ];
			transformed += morphTarget5 * morphTargetInfluences[ 5 ];
			transformed += morphTarget6 * morphTargetInfluences[ 6 ];
			transformed += morphTarget7 * morphTargetInfluences[ 7 ];
		#endif
	#endif
#endif`,zc=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#ifdef USE_NORMALMAP_TANGENTSPACE
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal, vNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 geometryNormal = normal;`,Bc=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Gc=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Vc=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,kc=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Hc=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Wc=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = geometryNormal;
#endif`,qc=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Xc=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Yc=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,jc=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha + 0.1;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Jc=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;
const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256., 256. );
const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
const float ShiftRight8 = 1. / 256.;
vec4 packDepthToRGBA( const in float v ) {
	vec4 r = vec4( fract( v * PackFactors ), v );
	r.yzw -= r.xyz * ShiftRight8;	return r * PackUpscale;
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors );
}
vec2 packDepthToRG( in highp float v ) {
	return packDepthToRGBA( v ).yx;
}
float unpackRGToDepth( const in highp vec2 v ) {
	return unpackRGBAToDepth( vec4( v.xy, 0.0, 0.0 ) );
}
vec4 pack2HalfToRGBA( vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,Zc=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Kc=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,$c=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Qc=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,th=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,eh=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,nh=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return shadow;
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
		vec3 lightToPosition = shadowCoord.xyz;
		float dp = ( length( lightToPosition ) - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );		dp += shadowBias;
		vec3 bd3D = normalize( lightToPosition );
		#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
			vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
			return (
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
			) * ( 1.0 / 9.0 );
		#else
			return texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
		#endif
	}
#endif`,ih=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,rh=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,sh=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,ah=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,oh=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	uniform int boneTextureSize;
	mat4 getBoneMatrix( const in float i ) {
		float j = i * 4.0;
		float x = mod( j, float( boneTextureSize ) );
		float y = floor( j / float( boneTextureSize ) );
		float dx = 1.0 / float( boneTextureSize );
		float dy = 1.0 / float( boneTextureSize );
		y = dy * ( y + 0.5 );
		vec4 v1 = texture2D( boneTexture, vec2( dx * ( x + 0.5 ), y ) );
		vec4 v2 = texture2D( boneTexture, vec2( dx * ( x + 1.5 ), y ) );
		vec4 v3 = texture2D( boneTexture, vec2( dx * ( x + 2.5 ), y ) );
		vec4 v4 = texture2D( boneTexture, vec2( dx * ( x + 3.5 ), y ) );
		mat4 bone = mat4( v1, v2, v3, v4 );
		return bone;
	}
#endif`,lh=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,ch=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,hh=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,uh=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,dh=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,fh=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return toneMappingExposure * color;
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 OptimizedCineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,ph=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmission = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmission.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmission.rgb, material.transmission );
#endif`,mh=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, vec2 fullSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		
		vec2 lodFudge = pow( 1.95, lod ) / fullSize;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec2 fullSize = vec2( textureSize( sampler, 0 ) );
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), fullSize, floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), fullSize, ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 applyVolumeAttenuation( const in vec3 radiance, const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return radiance;
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance * radiance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
		vec3 refractedRayExit = position + transmissionRay;
		vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
		vec2 refractionCoords = ndcPos.xy / ndcPos.w;
		refractionCoords += 1.0;
		refractionCoords /= 2.0;
		vec4 transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
		vec3 attenuatedColor = applyVolumeAttenuation( transmittedLight.rgb, length( transmissionRay ), attenuationColor, attenuationDistance );
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		return vec4( ( 1.0 - F ) * attenuatedColor * diffuseColor, transmittedLight.a );
	}
#endif`,gh=`#ifdef USE_UV
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,_h=`#ifdef USE_UV
	varying vec2 vUv;
#endif
#ifdef USE_UV2
	attribute vec2 uv2;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,vh=`#ifdef USE_UV
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,xh=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Mh=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Sh=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <encodings_fragment>
}`,yh=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,bh=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <encodings_fragment>
}`,wh=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Eh=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <encodings_fragment>
}`,Th=`#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <skinbase_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Ah=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( 1.0 );
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#endif
}`,Ch=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <skinbase_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Lh=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( 1.0 );
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Ph=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Rh=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <encodings_fragment>
}`,Dh=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Ih=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Uh=`#include <common>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Nh=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Oh=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Fh=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,zh=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Bh=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Gh=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Vh=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), opacity );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,kh=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Hh=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Wh=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,qh=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecular;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometry.clearcoatNormal, geometry.viewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + clearcoatSpecular * material.clearcoat;
	#endif
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Xh=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Yh=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,jh=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Jh=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Zh=`#include <common>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Kh=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
}`,$h=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );
	vec2 scale;
	scale.x = length( vec3( modelMatrix[ 0 ].x, modelMatrix[ 0 ].y, modelMatrix[ 0 ].z ) );
	scale.y = length( vec3( modelMatrix[ 1 ].x, modelMatrix[ 1 ].y, modelMatrix[ 1 ].z ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Qh=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
}`,Ct={alphamap_fragment:El,alphamap_pars_fragment:Tl,alphatest_fragment:Al,alphatest_pars_fragment:Cl,aomap_fragment:Ll,aomap_pars_fragment:Pl,begin_vertex:Rl,beginnormal_vertex:Dl,bsdfs:Il,iridescence_fragment:Ul,bumpmap_pars_fragment:Nl,clipping_planes_fragment:Ol,clipping_planes_pars_fragment:Fl,clipping_planes_pars_vertex:zl,clipping_planes_vertex:Bl,color_fragment:Gl,color_pars_fragment:Vl,color_pars_vertex:kl,color_vertex:Hl,common:Wl,cube_uv_reflection_fragment:ql,defaultnormal_vertex:Xl,displacementmap_pars_vertex:Yl,displacementmap_vertex:jl,emissivemap_fragment:Jl,emissivemap_pars_fragment:Zl,encodings_fragment:Kl,encodings_pars_fragment:$l,envmap_fragment:Ql,envmap_common_pars_fragment:tc,envmap_pars_fragment:ec,envmap_pars_vertex:nc,envmap_physical_pars_fragment:pc,envmap_vertex:ic,fog_vertex:rc,fog_pars_vertex:sc,fog_fragment:ac,fog_pars_fragment:oc,gradientmap_pars_fragment:lc,lightmap_fragment:cc,lightmap_pars_fragment:hc,lights_lambert_fragment:uc,lights_lambert_pars_fragment:dc,lights_pars_begin:fc,lights_toon_fragment:mc,lights_toon_pars_fragment:gc,lights_phong_fragment:_c,lights_phong_pars_fragment:vc,lights_physical_fragment:xc,lights_physical_pars_fragment:Mc,lights_fragment_begin:Sc,lights_fragment_maps:yc,lights_fragment_end:bc,logdepthbuf_fragment:wc,logdepthbuf_pars_fragment:Ec,logdepthbuf_pars_vertex:Tc,logdepthbuf_vertex:Ac,map_fragment:Cc,map_pars_fragment:Lc,map_particle_fragment:Pc,map_particle_pars_fragment:Rc,metalnessmap_fragment:Dc,metalnessmap_pars_fragment:Ic,morphcolor_vertex:Uc,morphnormal_vertex:Nc,morphtarget_pars_vertex:Oc,morphtarget_vertex:Fc,normal_fragment_begin:zc,normal_fragment_maps:Bc,normal_pars_fragment:Gc,normal_pars_vertex:Vc,normal_vertex:kc,normalmap_pars_fragment:Hc,clearcoat_normal_fragment_begin:Wc,clearcoat_normal_fragment_maps:qc,clearcoat_pars_fragment:Xc,iridescence_pars_fragment:Yc,output_fragment:jc,packing:Jc,premultiplied_alpha_fragment:Zc,project_vertex:Kc,dithering_fragment:$c,dithering_pars_fragment:Qc,roughnessmap_fragment:th,roughnessmap_pars_fragment:eh,shadowmap_pars_fragment:nh,shadowmap_pars_vertex:ih,shadowmap_vertex:rh,shadowmask_pars_fragment:sh,skinbase_vertex:ah,skinning_pars_vertex:oh,skinning_vertex:lh,skinnormal_vertex:ch,specularmap_fragment:hh,specularmap_pars_fragment:uh,tonemapping_fragment:dh,tonemapping_pars_fragment:fh,transmission_fragment:ph,transmission_pars_fragment:mh,uv_pars_fragment:gh,uv_pars_vertex:_h,uv_vertex:vh,worldpos_vertex:xh,background_vert:Mh,background_frag:Sh,backgroundCube_vert:yh,backgroundCube_frag:bh,cube_vert:wh,cube_frag:Eh,depth_vert:Th,depth_frag:Ah,distanceRGBA_vert:Ch,distanceRGBA_frag:Lh,equirect_vert:Ph,equirect_frag:Rh,linedashed_vert:Dh,linedashed_frag:Ih,meshbasic_vert:Uh,meshbasic_frag:Nh,meshlambert_vert:Oh,meshlambert_frag:Fh,meshmatcap_vert:zh,meshmatcap_frag:Bh,meshnormal_vert:Gh,meshnormal_frag:Vh,meshphong_vert:kh,meshphong_frag:Hh,meshphysical_vert:Wh,meshphysical_frag:qh,meshtoon_vert:Xh,meshtoon_frag:Yh,points_vert:jh,points_frag:Jh,shadow_vert:Zh,shadow_frag:Kh,sprite_vert:$h,sprite_frag:Qh},rt={common:{diffuse:{value:new Ot(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new It},alphaMap:{value:null},alphaMapTransform:{value:new It},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new It}},envmap:{envMap:{value:null},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new It}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new It}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new It},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new It},normalScale:{value:new dt(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new It},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new It}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new It}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new It}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Ot(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new Ot(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaTest:{value:0},uvTransform:{value:new It}},sprite:{diffuse:{value:new Ot(16777215)},opacity:{value:1},center:{value:new dt(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new It},alphaMap:{value:null},alphaTest:{value:0}}},Fe={basic:{uniforms:de([rt.common,rt.specularmap,rt.envmap,rt.aomap,rt.lightmap,rt.fog]),vertexShader:Ct.meshbasic_vert,fragmentShader:Ct.meshbasic_frag},lambert:{uniforms:de([rt.common,rt.specularmap,rt.envmap,rt.aomap,rt.lightmap,rt.emissivemap,rt.bumpmap,rt.normalmap,rt.displacementmap,rt.fog,rt.lights,{emissive:{value:new Ot(0)}}]),vertexShader:Ct.meshlambert_vert,fragmentShader:Ct.meshlambert_frag},phong:{uniforms:de([rt.common,rt.specularmap,rt.envmap,rt.aomap,rt.lightmap,rt.emissivemap,rt.bumpmap,rt.normalmap,rt.displacementmap,rt.fog,rt.lights,{emissive:{value:new Ot(0)},specular:{value:new Ot(1118481)},shininess:{value:30}}]),vertexShader:Ct.meshphong_vert,fragmentShader:Ct.meshphong_frag},standard:{uniforms:de([rt.common,rt.envmap,rt.aomap,rt.lightmap,rt.emissivemap,rt.bumpmap,rt.normalmap,rt.displacementmap,rt.roughnessmap,rt.metalnessmap,rt.fog,rt.lights,{emissive:{value:new Ot(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Ct.meshphysical_vert,fragmentShader:Ct.meshphysical_frag},toon:{uniforms:de([rt.common,rt.aomap,rt.lightmap,rt.emissivemap,rt.bumpmap,rt.normalmap,rt.displacementmap,rt.gradientmap,rt.fog,rt.lights,{emissive:{value:new Ot(0)}}]),vertexShader:Ct.meshtoon_vert,fragmentShader:Ct.meshtoon_frag},matcap:{uniforms:de([rt.common,rt.bumpmap,rt.normalmap,rt.displacementmap,rt.fog,{matcap:{value:null}}]),vertexShader:Ct.meshmatcap_vert,fragmentShader:Ct.meshmatcap_frag},points:{uniforms:de([rt.points,rt.fog]),vertexShader:Ct.points_vert,fragmentShader:Ct.points_frag},dashed:{uniforms:de([rt.common,rt.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Ct.linedashed_vert,fragmentShader:Ct.linedashed_frag},depth:{uniforms:de([rt.common,rt.displacementmap]),vertexShader:Ct.depth_vert,fragmentShader:Ct.depth_frag},normal:{uniforms:de([rt.common,rt.bumpmap,rt.normalmap,rt.displacementmap,{opacity:{value:1}}]),vertexShader:Ct.meshnormal_vert,fragmentShader:Ct.meshnormal_frag},sprite:{uniforms:de([rt.sprite,rt.fog]),vertexShader:Ct.sprite_vert,fragmentShader:Ct.sprite_frag},background:{uniforms:{uvTransform:{value:new It},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Ct.background_vert,fragmentShader:Ct.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1}},vertexShader:Ct.backgroundCube_vert,fragmentShader:Ct.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Ct.cube_vert,fragmentShader:Ct.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Ct.equirect_vert,fragmentShader:Ct.equirect_frag},distanceRGBA:{uniforms:de([rt.common,rt.displacementmap,{referencePosition:{value:new P},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Ct.distanceRGBA_vert,fragmentShader:Ct.distanceRGBA_frag},shadow:{uniforms:de([rt.lights,rt.fog,{color:{value:new Ot(0)},opacity:{value:1}}]),vertexShader:Ct.shadow_vert,fragmentShader:Ct.shadow_frag}};Fe.physical={uniforms:de([Fe.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new It},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new It},clearcoatNormalScale:{value:new dt(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new It},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new It},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new It},sheen:{value:0},sheenColor:{value:new Ot(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new It},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new It},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new It},transmissionSamplerSize:{value:new dt},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new It},attenuationDistance:{value:0},attenuationColor:{value:new Ot(0)},specularColor:{value:new Ot(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new It},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new It}}]),vertexShader:Ct.meshphysical_vert,fragmentShader:Ct.meshphysical_frag};const Ri={r:0,b:0,g:0};function tu(a,t,e,n,i,r,o){const s=new Ot(0);let c=r===!0?0:1,l,h,d=null,f=0,p=null;function g(m,u){let E=!1,x=u.isScene===!0?u.background:null;x&&x.isTexture&&(x=(u.backgroundBlurriness>0?e:t).get(x));const b=a.xr,y=b.getSession&&b.getSession();y&&y.environmentBlendMode==="additive"&&(x=null),x===null?_(s,c):x&&x.isColor&&(_(x,1),E=!0),(a.autoClear||E)&&a.clear(a.autoClearColor,a.autoClearDepth,a.autoClearStencil),x&&(x.isCubeTexture||x.mapping===Oi)?(h===void 0&&(h=new ze(new jn(1,1,1),new xn({name:"BackgroundCubeMaterial",uniforms:Yn(Fe.backgroundCube.uniforms),vertexShader:Fe.backgroundCube.vertexShader,fragmentShader:Fe.backgroundCube.fragmentShader,side:pe,depthTest:!1,depthWrite:!1,fog:!1})),h.geometry.deleteAttribute("normal"),h.geometry.deleteAttribute("uv"),h.onBeforeRender=function(C,D,R){this.matrixWorld.copyPosition(R.matrixWorld)},Object.defineProperty(h.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(h)),h.material.uniforms.envMap.value=x,h.material.uniforms.flipEnvMap.value=x.isCubeTexture&&x.isRenderTargetTexture===!1?-1:1,h.material.uniforms.backgroundBlurriness.value=u.backgroundBlurriness,h.material.uniforms.backgroundIntensity.value=u.backgroundIntensity,h.material.toneMapped=x.encoding!==Gt,(d!==x||f!==x.version||p!==a.toneMapping)&&(h.material.needsUpdate=!0,d=x,f=x.version,p=a.toneMapping),h.layers.enableAll(),m.unshift(h,h.geometry,h.material,0,0,null)):x&&x.isTexture&&(l===void 0&&(l=new ze(new zi(2,2),new xn({name:"BackgroundMaterial",uniforms:Yn(Fe.background.uniforms),vertexShader:Fe.background.vertexShader,fragmentShader:Fe.background.fragmentShader,side:en,depthTest:!1,depthWrite:!1,fog:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(l)),l.material.uniforms.t2D.value=x,l.material.uniforms.backgroundIntensity.value=u.backgroundIntensity,l.material.toneMapped=x.encoding!==Gt,x.matrixAutoUpdate===!0&&x.updateMatrix(),l.material.uniforms.uvTransform.value.copy(x.matrix),(d!==x||f!==x.version||p!==a.toneMapping)&&(l.material.needsUpdate=!0,d=x,f=x.version,p=a.toneMapping),l.layers.enableAll(),m.unshift(l,l.geometry,l.material,0,0,null))}function _(m,u){m.getRGB(Ri,za(a)),n.buffers.color.setClear(Ri.r,Ri.g,Ri.b,u,o)}return{getClearColor:function(){return s},setClearColor:function(m,u=1){s.set(m),c=u,_(s,c)},getClearAlpha:function(){return c},setClearAlpha:function(m){c=m,_(s,c)},render:g}}function eu(a,t,e,n){const i=a.getParameter(34921),r=n.isWebGL2?null:t.get("OES_vertex_array_object"),o=n.isWebGL2||r!==null,s={},c=m(null);let l=c,h=!1;function d(N,B,Q,et,j){let Z=!1;if(o){const nt=_(et,Q,B);l!==nt&&(l=nt,p(l.object)),Z=u(N,et,Q,j),Z&&E(N,et,Q,j)}else{const nt=B.wireframe===!0;(l.geometry!==et.id||l.program!==Q.id||l.wireframe!==nt)&&(l.geometry=et.id,l.program=Q.id,l.wireframe=nt,Z=!0)}j!==null&&e.update(j,34963),(Z||h)&&(h=!1,R(N,B,Q,et),j!==null&&a.bindBuffer(34963,e.get(j).buffer))}function f(){return n.isWebGL2?a.createVertexArray():r.createVertexArrayOES()}function p(N){return n.isWebGL2?a.bindVertexArray(N):r.bindVertexArrayOES(N)}function g(N){return n.isWebGL2?a.deleteVertexArray(N):r.deleteVertexArrayOES(N)}function _(N,B,Q){const et=Q.wireframe===!0;let j=s[N.id];j===void 0&&(j={},s[N.id]=j);let Z=j[B.id];Z===void 0&&(Z={},j[B.id]=Z);let nt=Z[et];return nt===void 0&&(nt=m(f()),Z[et]=nt),nt}function m(N){const B=[],Q=[],et=[];for(let j=0;j<i;j++)B[j]=0,Q[j]=0,et[j]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:B,enabledAttributes:Q,attributeDivisors:et,object:N,attributes:{},index:null}}function u(N,B,Q,et){const j=l.attributes,Z=B.attributes;let nt=0;const vt=Q.getAttributes();for(const ct in vt)if(vt[ct].location>=0){const J=j[ct];let st=Z[ct];if(st===void 0&&(ct==="instanceMatrix"&&N.instanceMatrix&&(st=N.instanceMatrix),ct==="instanceColor"&&N.instanceColor&&(st=N.instanceColor)),J===void 0||J.attribute!==st||st&&J.data!==st.data)return!0;nt++}return l.attributesNum!==nt||l.index!==et}function E(N,B,Q,et){const j={},Z=B.attributes;let nt=0;const vt=Q.getAttributes();for(const ct in vt)if(vt[ct].location>=0){let J=Z[ct];J===void 0&&(ct==="instanceMatrix"&&N.instanceMatrix&&(J=N.instanceMatrix),ct==="instanceColor"&&N.instanceColor&&(J=N.instanceColor));const st={};st.attribute=J,J&&J.data&&(st.data=J.data),j[ct]=st,nt++}l.attributes=j,l.attributesNum=nt,l.index=et}function x(){const N=l.newAttributes;for(let B=0,Q=N.length;B<Q;B++)N[B]=0}function b(N){y(N,0)}function y(N,B){const Q=l.newAttributes,et=l.enabledAttributes,j=l.attributeDivisors;Q[N]=1,et[N]===0&&(a.enableVertexAttribArray(N),et[N]=1),j[N]!==B&&((n.isWebGL2?a:t.get("ANGLE_instanced_arrays"))[n.isWebGL2?"vertexAttribDivisor":"vertexAttribDivisorANGLE"](N,B),j[N]=B)}function C(){const N=l.newAttributes,B=l.enabledAttributes;for(let Q=0,et=B.length;Q<et;Q++)B[Q]!==N[Q]&&(a.disableVertexAttribArray(Q),B[Q]=0)}function D(N,B,Q,et,j,Z){n.isWebGL2===!0&&(Q===5124||Q===5125)?a.vertexAttribIPointer(N,B,Q,j,Z):a.vertexAttribPointer(N,B,Q,et,j,Z)}function R(N,B,Q,et){if(n.isWebGL2===!1&&(N.isInstancedMesh||et.isInstancedBufferGeometry)&&t.get("ANGLE_instanced_arrays")===null)return;x();const j=et.attributes,Z=Q.getAttributes(),nt=B.defaultAttributeValues;for(const vt in Z){const ct=Z[vt];if(ct.location>=0){let V=j[vt];if(V===void 0&&(vt==="instanceMatrix"&&N.instanceMatrix&&(V=N.instanceMatrix),vt==="instanceColor"&&N.instanceColor&&(V=N.instanceColor)),V!==void 0){const J=V.normalized,st=V.itemSize,lt=e.get(V);if(lt===void 0)continue;const z=lt.buffer,bt=lt.type,St=lt.bytesPerElement;if(V.isInterleavedBufferAttribute){const it=V.data,Mt=it.stride,Ft=V.offset;if(it.isInstancedInterleavedBuffer){for(let gt=0;gt<ct.locationSize;gt++)y(ct.location+gt,it.meshPerAttribute);N.isInstancedMesh!==!0&&et._maxInstanceCount===void 0&&(et._maxInstanceCount=it.meshPerAttribute*it.count)}else for(let gt=0;gt<ct.locationSize;gt++)b(ct.location+gt);a.bindBuffer(34962,z);for(let gt=0;gt<ct.locationSize;gt++)D(ct.location+gt,st/ct.locationSize,bt,J,Mt*St,(Ft+st/ct.locationSize*gt)*St)}else{if(V.isInstancedBufferAttribute){for(let it=0;it<ct.locationSize;it++)y(ct.location+it,V.meshPerAttribute);N.isInstancedMesh!==!0&&et._maxInstanceCount===void 0&&(et._maxInstanceCount=V.meshPerAttribute*V.count)}else for(let it=0;it<ct.locationSize;it++)b(ct.location+it);a.bindBuffer(34962,z);for(let it=0;it<ct.locationSize;it++)D(ct.location+it,st/ct.locationSize,bt,J,st*St,st/ct.locationSize*it*St)}}else if(nt!==void 0){const J=nt[vt];if(J!==void 0)switch(J.length){case 2:a.vertexAttrib2fv(ct.location,J);break;case 3:a.vertexAttrib3fv(ct.location,J);break;case 4:a.vertexAttrib4fv(ct.location,J);break;default:a.vertexAttrib1fv(ct.location,J)}}}}C()}function M(){q();for(const N in s){const B=s[N];for(const Q in B){const et=B[Q];for(const j in et)g(et[j].object),delete et[j];delete B[Q]}delete s[N]}}function T(N){if(s[N.id]===void 0)return;const B=s[N.id];for(const Q in B){const et=B[Q];for(const j in et)g(et[j].object),delete et[j];delete B[Q]}delete s[N.id]}function X(N){for(const B in s){const Q=s[B];if(Q[N.id]===void 0)continue;const et=Q[N.id];for(const j in et)g(et[j].object),delete et[j];delete Q[N.id]}}function q(){I(),h=!0,l!==c&&(l=c,p(l.object))}function I(){c.geometry=null,c.program=null,c.wireframe=!1}return{setup:d,reset:q,resetDefaultState:I,dispose:M,releaseStatesOfGeometry:T,releaseStatesOfProgram:X,initAttributes:x,enableAttribute:b,disableUnusedAttributes:C}}function nu(a,t,e,n){const i=n.isWebGL2;let r;function o(l){r=l}function s(l,h){a.drawArrays(r,l,h),e.update(h,r,1)}function c(l,h,d){if(d===0)return;let f,p;if(i)f=a,p="drawArraysInstanced";else if(f=t.get("ANGLE_instanced_arrays"),p="drawArraysInstancedANGLE",f===null){console.error("THREE.WebGLBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");return}f[p](r,l,h,d),e.update(h,r,d)}this.setMode=o,this.render=s,this.renderInstances=c}function iu(a,t,e){let n;function i(){if(n!==void 0)return n;if(t.has("EXT_texture_filter_anisotropic")===!0){const D=t.get("EXT_texture_filter_anisotropic");n=a.getParameter(D.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(D){if(D==="highp"){if(a.getShaderPrecisionFormat(35633,36338).precision>0&&a.getShaderPrecisionFormat(35632,36338).precision>0)return"highp";D="mediump"}return D==="mediump"&&a.getShaderPrecisionFormat(35633,36337).precision>0&&a.getShaderPrecisionFormat(35632,36337).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&a.constructor.name==="WebGL2RenderingContext";let s=e.precision!==void 0?e.precision:"highp";const c=r(s);c!==s&&(console.warn("THREE.WebGLRenderer:",s,"not supported, using",c,"instead."),s=c);const l=o||t.has("WEBGL_draw_buffers"),h=e.logarithmicDepthBuffer===!0,d=a.getParameter(34930),f=a.getParameter(35660),p=a.getParameter(3379),g=a.getParameter(34076),_=a.getParameter(34921),m=a.getParameter(36347),u=a.getParameter(36348),E=a.getParameter(36349),x=f>0,b=o||t.has("OES_texture_float"),y=x&&b,C=o?a.getParameter(36183):0;return{isWebGL2:o,drawBuffers:l,getMaxAnisotropy:i,getMaxPrecision:r,precision:s,logarithmicDepthBuffer:h,maxTextures:d,maxVertexTextures:f,maxTextureSize:p,maxCubemapSize:g,maxAttributes:_,maxVertexUniforms:m,maxVaryings:u,maxFragmentUniforms:E,vertexTextures:x,floatFragmentTextures:b,floatVertexTextures:y,maxSamples:C}}function ru(a){const t=this;let e=null,n=0,i=!1,r=!1;const o=new cn,s=new It,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(d,f){const p=d.length!==0||f||n!==0||i;return i=f,n=d.length,p},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(d,f){e=h(d,f,0)},this.setState=function(d,f,p){const g=d.clippingPlanes,_=d.clipIntersection,m=d.clipShadows,u=a.get(d);if(!i||g===null||g.length===0||r&&!m)r?h(null):l();else{const E=r?0:n,x=E*4;let b=u.clippingState||null;c.value=b,b=h(g,f,x,p);for(let y=0;y!==x;++y)b[y]=e[y];u.clippingState=b,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=E}};function l(){c.value!==e&&(c.value=e,c.needsUpdate=n>0),t.numPlanes=n,t.numIntersection=0}function h(d,f,p,g){const _=d!==null?d.length:0;let m=null;if(_!==0){if(m=c.value,g!==!0||m===null){const u=p+_*4,E=f.matrixWorldInverse;s.getNormalMatrix(E),(m===null||m.length<u)&&(m=new Float32Array(u));for(let x=0,b=p;x!==_;++x,b+=4)o.copy(d[x]).applyMatrix4(E,s),o.normal.toArray(m,b),m[b+3]=o.constant}c.value=m,c.needsUpdate=!0}return t.numPlanes=_,t.numIntersection=0,m}}function su(a){let t=new WeakMap;function e(o,s){return s===Ar?o.mapping=Wn:s===Cr&&(o.mapping=qn),o}function n(o){if(o&&o.isTexture&&o.isRenderTargetTexture===!1){const s=o.mapping;if(s===Ar||s===Cr)if(t.has(o)){const c=t.get(o).texture;return e(c,o.mapping)}else{const c=o.image;if(c&&c.height>0){const l=new Sl(c.height/2);return l.fromEquirectangularTexture(a,o),t.set(o,l),o.addEventListener("dispose",i),e(l.texture,o.mapping)}else return null}}return o}function i(o){const s=o.target;s.removeEventListener("dispose",i);const c=t.get(s);c!==void 0&&(t.delete(s),c.dispose())}function r(){t=new WeakMap}return{get:n,dispose:r}}class ka extends Ba{constructor(t=-1,e=1,n=1,i=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=i,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,n,i,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,i=(this.top+this.bottom)/2;let r=n-t,o=n+t,s=i+e,c=i-e;if(this.view!==null&&this.view.enabled){const l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,o=r+l*this.view.width,s-=h*this.view.offsetY,c=s-h*this.view.height}this.projectionMatrix.makeOrthographic(r,o,s,c,this.near,this.far),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}}const Gn=4,Hs=[.125,.215,.35,.446,.526,.582],un=20,gr=new ka,Ws=new Ot;let _r=null;const hn=(1+Math.sqrt(5))/2,zn=1/hn,qs=[new P(1,1,1),new P(-1,1,1),new P(1,1,-1),new P(-1,1,-1),new P(0,hn,zn),new P(0,hn,-zn),new P(zn,0,hn),new P(-zn,0,hn),new P(hn,zn,0),new P(-hn,zn,0)];class Xs{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,n=.1,i=100){_r=this._renderer.getRenderTarget(),this._setSize(256);const r=this._allocateTargets();return r.depthBuffer=!0,this._sceneToCubeUV(t,n,i,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Js(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=js(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(t){this._lodMax=Math.floor(Math.log2(t)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let t=0;t<this._lodPlanes.length;t++)this._lodPlanes[t].dispose()}_cleanup(t){this._renderer.setRenderTarget(_r),t.scissorTest=!1,Di(t,0,0,t.width,t.height)}_fromTexture(t,e){t.mapping===Wn||t.mapping===qn?this._setSize(t.image.length===0?16:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4),_r=this._renderer.getRenderTarget();const n=e||this._allocateTargets();return this._textureToCubeUV(t,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const t=3*Math.max(this._cubeSize,112),e=4*this._cubeSize,n={magFilter:be,minFilter:be,generateMipmaps:!1,type:ai,format:De,encoding:gn,depthBuffer:!1},i=Ys(t,e,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==t||this._pingPongRenderTarget.height!==e){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Ys(t,e,n);const{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=au(r)),this._blurMaterial=ou(r,t,e)}return i}_compileMaterial(t){const e=new ze(this._lodPlanes[0],t);this._renderer.compile(e,gr)}_sceneToCubeUV(t,e,n,i){const s=new we(90,1,e,n),c=[1,-1,1,1,1,1],l=[1,1,1,-1,-1,-1],h=this._renderer,d=h.autoClear,f=h.toneMapping;h.getClearColor(Ws),h.toneMapping=je,h.autoClear=!1;const p=new Na({name:"PMREM.Background",side:pe,depthWrite:!1,depthTest:!1}),g=new ze(new jn,p);let _=!1;const m=t.background;m?m.isColor&&(p.color.copy(m),t.background=null,_=!0):(p.color.copy(Ws),_=!0);for(let u=0;u<6;u++){const E=u%3;E===0?(s.up.set(0,c[u],0),s.lookAt(l[u],0,0)):E===1?(s.up.set(0,0,c[u]),s.lookAt(0,l[u],0)):(s.up.set(0,c[u],0),s.lookAt(0,0,l[u]));const x=this._cubeSize;Di(i,E*x,u>2?x:0,x,x),h.setRenderTarget(i),_&&h.render(g,s),h.render(t,s)}g.geometry.dispose(),g.material.dispose(),h.toneMapping=f,h.autoClear=d,t.background=m}_textureToCubeUV(t,e){const n=this._renderer,i=t.mapping===Wn||t.mapping===qn;i?(this._cubemapMaterial===null&&(this._cubemapMaterial=Js()),this._cubemapMaterial.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=js());const r=i?this._cubemapMaterial:this._equirectMaterial,o=new ze(this._lodPlanes[0],r),s=r.uniforms;s.envMap.value=t;const c=this._cubeSize;Di(e,0,0,3*c,2*c),n.setRenderTarget(e),n.render(o,gr)}_applyPMREM(t){const e=this._renderer,n=e.autoClear;e.autoClear=!1;for(let i=1;i<this._lodPlanes.length;i++){const r=Math.sqrt(this._sigmas[i]*this._sigmas[i]-this._sigmas[i-1]*this._sigmas[i-1]),o=qs[(i-1)%qs.length];this._blur(t,i-1,i,r,o)}e.autoClear=n}_blur(t,e,n,i,r){const o=this._pingPongRenderTarget;this._halfBlur(t,o,e,n,i,"latitudinal",r),this._halfBlur(o,t,n,n,i,"longitudinal",r)}_halfBlur(t,e,n,i,r,o,s){const c=this._renderer,l=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const h=3,d=new ze(this._lodPlanes[i],l),f=l.uniforms,p=this._sizeLods[n]-1,g=isFinite(r)?Math.PI/(2*p):2*Math.PI/(2*un-1),_=r/g,m=isFinite(r)?1+Math.floor(h*_):un;m>un&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${un}`);const u=[];let E=0;for(let D=0;D<un;++D){const R=D/_,M=Math.exp(-R*R/2);u.push(M),D===0?E+=M:D<m&&(E+=2*M)}for(let D=0;D<u.length;D++)u[D]=u[D]/E;f.envMap.value=t.texture,f.samples.value=m,f.weights.value=u,f.latitudinal.value=o==="latitudinal",s&&(f.poleAxis.value=s);const{_lodMax:x}=this;f.dTheta.value=g,f.mipInt.value=x-n;const b=this._sizeLods[i],y=3*b*(i>x-Gn?i-x+Gn:0),C=4*(this._cubeSize-b);Di(e,y,C,3*b,2*b),c.setRenderTarget(e),c.render(d,gr)}}function au(a){const t=[],e=[],n=[];let i=a;const r=a-Gn+1+Hs.length;for(let o=0;o<r;o++){const s=Math.pow(2,i);e.push(s);let c=1/s;o>a-Gn?c=Hs[o-a+Gn-1]:o===0&&(c=0),n.push(c);const l=1/(s-2),h=-l,d=1+l,f=[h,h,d,h,d,d,h,h,d,d,h,d],p=6,g=6,_=3,m=2,u=1,E=new Float32Array(_*g*p),x=new Float32Array(m*g*p),b=new Float32Array(u*g*p);for(let C=0;C<p;C++){const D=C%3*2/3-1,R=C>2?0:-1,M=[D,R,0,D+2/3,R,0,D+2/3,R+1,0,D,R,0,D+2/3,R+1,0,D,R+1,0];E.set(M,_*g*C),x.set(f,m*g*C);const T=[C,C,C,C,C,C];b.set(T,u*g*C)}const y=new Ee;y.setAttribute("position",new Ie(E,_)),y.setAttribute("uv",new Ie(x,m)),y.setAttribute("faceIndex",new Ie(b,u)),t.push(y),i>Gn&&i--}return{lodPlanes:t,sizeLods:e,sigmas:n}}function Ys(a,t,e){const n=new _n(a,t,e);return n.texture.mapping=Oi,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Di(a,t,e,n,i){a.viewport.set(t,e,n,i),a.scissor.set(t,e,n,i)}function ou(a,t,e){const n=new Float32Array(un),i=new P(0,1,0);return new xn({name:"SphericalGaussianBlur",defines:{n:un,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/e,CUBEUV_MAX_MIP:`${a}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:i}},vertexShader:zr(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:tn,depthTest:!1,depthWrite:!1})}function js(){return new xn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:zr(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:tn,depthTest:!1,depthWrite:!1})}function Js(){return new xn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:zr(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:tn,depthTest:!1,depthWrite:!1})}function zr(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function lu(a){let t=new WeakMap,e=null;function n(s){if(s&&s.isTexture){const c=s.mapping,l=c===Ar||c===Cr,h=c===Wn||c===qn;if(l||h)if(s.isRenderTargetTexture&&s.needsPMREMUpdate===!0){s.needsPMREMUpdate=!1;let d=t.get(s);return e===null&&(e=new Xs(a)),d=l?e.fromEquirectangular(s,d):e.fromCubemap(s,d),t.set(s,d),d.texture}else{if(t.has(s))return t.get(s).texture;{const d=s.image;if(l&&d&&d.height>0||h&&d&&i(d)){e===null&&(e=new Xs(a));const f=l?e.fromEquirectangular(s):e.fromCubemap(s);return t.set(s,f),s.addEventListener("dispose",r),f.texture}else return null}}}return s}function i(s){let c=0;const l=6;for(let h=0;h<l;h++)s[h]!==void 0&&c++;return c===l}function r(s){const c=s.target;c.removeEventListener("dispose",r);const l=t.get(c);l!==void 0&&(t.delete(c),l.dispose())}function o(){t=new WeakMap,e!==null&&(e.dispose(),e=null)}return{get:n,dispose:o}}function cu(a){const t={};function e(n){if(t[n]!==void 0)return t[n];let i;switch(n){case"WEBGL_depth_texture":i=a.getExtension("WEBGL_depth_texture")||a.getExtension("MOZ_WEBGL_depth_texture")||a.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":i=a.getExtension("EXT_texture_filter_anisotropic")||a.getExtension("MOZ_EXT_texture_filter_anisotropic")||a.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":i=a.getExtension("WEBGL_compressed_texture_s3tc")||a.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||a.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":i=a.getExtension("WEBGL_compressed_texture_pvrtc")||a.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:i=a.getExtension(n)}return t[n]=i,i}return{has:function(n){return e(n)!==null},init:function(n){n.isWebGL2?e("EXT_color_buffer_float"):(e("WEBGL_depth_texture"),e("OES_texture_float"),e("OES_texture_half_float"),e("OES_texture_half_float_linear"),e("OES_standard_derivatives"),e("OES_element_index_uint"),e("OES_vertex_array_object"),e("ANGLE_instanced_arrays")),e("OES_texture_float_linear"),e("EXT_color_buffer_half_float"),e("WEBGL_multisampled_render_to_texture")},get:function(n){const i=e(n);return i===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),i}}}function hu(a,t,e,n){const i={},r=new WeakMap;function o(d){const f=d.target;f.index!==null&&t.remove(f.index);for(const g in f.attributes)t.remove(f.attributes[g]);f.removeEventListener("dispose",o),delete i[f.id];const p=r.get(f);p&&(t.remove(p),r.delete(f)),n.releaseStatesOfGeometry(f),f.isInstancedBufferGeometry===!0&&delete f._maxInstanceCount,e.memory.geometries--}function s(d,f){return i[f.id]===!0||(f.addEventListener("dispose",o),i[f.id]=!0,e.memory.geometries++),f}function c(d){const f=d.attributes;for(const g in f)t.update(f[g],34962);const p=d.morphAttributes;for(const g in p){const _=p[g];for(let m=0,u=_.length;m<u;m++)t.update(_[m],34962)}}function l(d){const f=[],p=d.index,g=d.attributes.position;let _=0;if(p!==null){const E=p.array;_=p.version;for(let x=0,b=E.length;x<b;x+=3){const y=E[x+0],C=E[x+1],D=E[x+2];f.push(y,C,C,D,D,y)}}else{const E=g.array;_=g.version;for(let x=0,b=E.length/3-1;x<b;x+=3){const y=x+0,C=x+1,D=x+2;f.push(y,C,C,D,D,y)}}const m=new(La(f)?Fa:Oa)(f,1);m.version=_;const u=r.get(d);u&&t.remove(u),r.set(d,m)}function h(d){const f=r.get(d);if(f){const p=d.index;p!==null&&f.version<p.version&&l(d)}else l(d);return r.get(d)}return{get:s,update:c,getWireframeAttribute:h}}function uu(a,t,e,n){const i=n.isWebGL2;let r;function o(f){r=f}let s,c;function l(f){s=f.type,c=f.bytesPerElement}function h(f,p){a.drawElements(r,p,s,f*c),e.update(p,r,1)}function d(f,p,g){if(g===0)return;let _,m;if(i)_=a,m="drawElementsInstanced";else if(_=t.get("ANGLE_instanced_arrays"),m="drawElementsInstancedANGLE",_===null){console.error("THREE.WebGLIndexedBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");return}_[m](r,p,s,f*c,g),e.update(p,r,g)}this.setMode=o,this.setIndex=l,this.render=h,this.renderInstances=d}function du(a){const t={geometries:0,textures:0},e={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,s){switch(e.calls++,o){case 4:e.triangles+=s*(r/3);break;case 1:e.lines+=s*(r/2);break;case 3:e.lines+=s*(r-1);break;case 2:e.lines+=s*r;break;case 0:e.points+=s*r;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",o);break}}function i(){e.frame++,e.calls=0,e.triangles=0,e.points=0,e.lines=0}return{memory:t,render:e,programs:null,autoReset:!0,reset:i,update:n}}function fu(a,t){return a[0]-t[0]}function pu(a,t){return Math.abs(t[1])-Math.abs(a[1])}function mu(a,t,e){const n={},i=new Float32Array(8),r=new WeakMap,o=new ie,s=[];for(let l=0;l<8;l++)s[l]=[l,0];function c(l,h,d){const f=l.morphTargetInfluences;if(t.isWebGL2===!0){const p=h.morphAttributes.position||h.morphAttributes.normal||h.morphAttributes.color,g=p!==void 0?p.length:0;let _=r.get(h);if(_===void 0||_.count!==g){let N=function(){q.dispose(),r.delete(h),h.removeEventListener("dispose",N)};_!==void 0&&_.texture.dispose();const E=h.morphAttributes.position!==void 0,x=h.morphAttributes.normal!==void 0,b=h.morphAttributes.color!==void 0,y=h.morphAttributes.position||[],C=h.morphAttributes.normal||[],D=h.morphAttributes.color||[];let R=0;E===!0&&(R=1),x===!0&&(R=2),b===!0&&(R=3);let M=h.attributes.position.count*R,T=1;M>t.maxTextureSize&&(T=Math.ceil(M/t.maxTextureSize),M=t.maxTextureSize);const X=new Float32Array(M*T*4*g),q=new Da(X,M,T,g);q.type=fn,q.needsUpdate=!0;const I=R*4;for(let B=0;B<g;B++){const Q=y[B],et=C[B],j=D[B],Z=M*T*4*B;for(let nt=0;nt<Q.count;nt++){const vt=nt*I;E===!0&&(o.fromBufferAttribute(Q,nt),X[Z+vt+0]=o.x,X[Z+vt+1]=o.y,X[Z+vt+2]=o.z,X[Z+vt+3]=0),x===!0&&(o.fromBufferAttribute(et,nt),X[Z+vt+4]=o.x,X[Z+vt+5]=o.y,X[Z+vt+6]=o.z,X[Z+vt+7]=0),b===!0&&(o.fromBufferAttribute(j,nt),X[Z+vt+8]=o.x,X[Z+vt+9]=o.y,X[Z+vt+10]=o.z,X[Z+vt+11]=j.itemSize===4?o.w:1)}}_={count:g,texture:q,size:new dt(M,T)},r.set(h,_),h.addEventListener("dispose",N)}let m=0;for(let E=0;E<f.length;E++)m+=f[E];const u=h.morphTargetsRelative?1:1-m;d.getUniforms().setValue(a,"morphTargetBaseInfluence",u),d.getUniforms().setValue(a,"morphTargetInfluences",f),d.getUniforms().setValue(a,"morphTargetsTexture",_.texture,e),d.getUniforms().setValue(a,"morphTargetsTextureSize",_.size)}else{const p=f===void 0?0:f.length;let g=n[h.id];if(g===void 0||g.length!==p){g=[];for(let x=0;x<p;x++)g[x]=[x,0];n[h.id]=g}for(let x=0;x<p;x++){const b=g[x];b[0]=x,b[1]=f[x]}g.sort(pu);for(let x=0;x<8;x++)x<p&&g[x][1]?(s[x][0]=g[x][0],s[x][1]=g[x][1]):(s[x][0]=Number.MAX_SAFE_INTEGER,s[x][1]=0);s.sort(fu);const _=h.morphAttributes.position,m=h.morphAttributes.normal;let u=0;for(let x=0;x<8;x++){const b=s[x],y=b[0],C=b[1];y!==Number.MAX_SAFE_INTEGER&&C?(_&&h.getAttribute("morphTarget"+x)!==_[y]&&h.setAttribute("morphTarget"+x,_[y]),m&&h.getAttribute("morphNormal"+x)!==m[y]&&h.setAttribute("morphNormal"+x,m[y]),i[x]=C,u+=C):(_&&h.hasAttribute("morphTarget"+x)===!0&&h.deleteAttribute("morphTarget"+x),m&&h.hasAttribute("morphNormal"+x)===!0&&h.deleteAttribute("morphNormal"+x),i[x]=0)}const E=h.morphTargetsRelative?1:1-u;d.getUniforms().setValue(a,"morphTargetBaseInfluence",E),d.getUniforms().setValue(a,"morphTargetInfluences",i)}}return{update:c}}function gu(a,t,e,n){let i=new WeakMap;function r(c){const l=n.render.frame,h=c.geometry,d=t.get(c,h);return i.get(d)!==l&&(t.update(d),i.set(d,l)),c.isInstancedMesh&&(c.hasEventListener("dispose",s)===!1&&c.addEventListener("dispose",s),e.update(c.instanceMatrix,34962),c.instanceColor!==null&&e.update(c.instanceColor,34962)),d}function o(){i=new WeakMap}function s(c){const l=c.target;l.removeEventListener("dispose",s),e.remove(l.instanceMatrix),l.instanceColor!==null&&e.remove(l.instanceColor)}return{update:r,dispose:o}}const Ha=new Me,Wa=new Da,qa=new sl,Xa=new Ga,Zs=[],Ks=[],$s=new Float32Array(16),Qs=new Float32Array(9),ta=new Float32Array(4);function Jn(a,t,e){const n=a[0];if(n<=0||n>0)return a;const i=t*e;let r=Zs[i];if(r===void 0&&(r=new Float32Array(i),Zs[i]=r),t!==0){n.toArray(r,0);for(let o=1,s=0;o!==t;++o)s+=e,a[o].toArray(r,s)}return r}function $t(a,t){if(a.length!==t.length)return!1;for(let e=0,n=a.length;e<n;e++)if(a[e]!==t[e])return!1;return!0}function Qt(a,t){for(let e=0,n=t.length;e<n;e++)a[e]=t[e]}function Bi(a,t){let e=Ks[t];e===void 0&&(e=new Int32Array(t),Ks[t]=e);for(let n=0;n!==t;++n)e[n]=a.allocateTextureUnit();return e}function _u(a,t){const e=this.cache;e[0]!==t&&(a.uniform1f(this.addr,t),e[0]=t)}function vu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(a.uniform2f(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if($t(e,t))return;a.uniform2fv(this.addr,t),Qt(e,t)}}function xu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(a.uniform3f(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else if(t.r!==void 0)(e[0]!==t.r||e[1]!==t.g||e[2]!==t.b)&&(a.uniform3f(this.addr,t.r,t.g,t.b),e[0]=t.r,e[1]=t.g,e[2]=t.b);else{if($t(e,t))return;a.uniform3fv(this.addr,t),Qt(e,t)}}function Mu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(a.uniform4f(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if($t(e,t))return;a.uniform4fv(this.addr,t),Qt(e,t)}}function Su(a,t){const e=this.cache,n=t.elements;if(n===void 0){if($t(e,t))return;a.uniformMatrix2fv(this.addr,!1,t),Qt(e,t)}else{if($t(e,n))return;ta.set(n),a.uniformMatrix2fv(this.addr,!1,ta),Qt(e,n)}}function yu(a,t){const e=this.cache,n=t.elements;if(n===void 0){if($t(e,t))return;a.uniformMatrix3fv(this.addr,!1,t),Qt(e,t)}else{if($t(e,n))return;Qs.set(n),a.uniformMatrix3fv(this.addr,!1,Qs),Qt(e,n)}}function bu(a,t){const e=this.cache,n=t.elements;if(n===void 0){if($t(e,t))return;a.uniformMatrix4fv(this.addr,!1,t),Qt(e,t)}else{if($t(e,n))return;$s.set(n),a.uniformMatrix4fv(this.addr,!1,$s),Qt(e,n)}}function wu(a,t){const e=this.cache;e[0]!==t&&(a.uniform1i(this.addr,t),e[0]=t)}function Eu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(a.uniform2i(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if($t(e,t))return;a.uniform2iv(this.addr,t),Qt(e,t)}}function Tu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(a.uniform3i(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if($t(e,t))return;a.uniform3iv(this.addr,t),Qt(e,t)}}function Au(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(a.uniform4i(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if($t(e,t))return;a.uniform4iv(this.addr,t),Qt(e,t)}}function Cu(a,t){const e=this.cache;e[0]!==t&&(a.uniform1ui(this.addr,t),e[0]=t)}function Lu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(a.uniform2ui(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if($t(e,t))return;a.uniform2uiv(this.addr,t),Qt(e,t)}}function Pu(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(a.uniform3ui(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if($t(e,t))return;a.uniform3uiv(this.addr,t),Qt(e,t)}}function Ru(a,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(a.uniform4ui(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if($t(e,t))return;a.uniform4uiv(this.addr,t),Qt(e,t)}}function Du(a,t,e){const n=this.cache,i=e.allocateTextureUnit();n[0]!==i&&(a.uniform1i(this.addr,i),n[0]=i),e.setTexture2D(t||Ha,i)}function Iu(a,t,e){const n=this.cache,i=e.allocateTextureUnit();n[0]!==i&&(a.uniform1i(this.addr,i),n[0]=i),e.setTexture3D(t||qa,i)}function Uu(a,t,e){const n=this.cache,i=e.allocateTextureUnit();n[0]!==i&&(a.uniform1i(this.addr,i),n[0]=i),e.setTextureCube(t||Xa,i)}function Nu(a,t,e){const n=this.cache,i=e.allocateTextureUnit();n[0]!==i&&(a.uniform1i(this.addr,i),n[0]=i),e.setTexture2DArray(t||Wa,i)}function Ou(a){switch(a){case 5126:return _u;case 35664:return vu;case 35665:return xu;case 35666:return Mu;case 35674:return Su;case 35675:return yu;case 35676:return bu;case 5124:case 35670:return wu;case 35667:case 35671:return Eu;case 35668:case 35672:return Tu;case 35669:case 35673:return Au;case 5125:return Cu;case 36294:return Lu;case 36295:return Pu;case 36296:return Ru;case 35678:case 36198:case 36298:case 36306:case 35682:return Du;case 35679:case 36299:case 36307:return Iu;case 35680:case 36300:case 36308:case 36293:return Uu;case 36289:case 36303:case 36311:case 36292:return Nu}}function Fu(a,t){a.uniform1fv(this.addr,t)}function zu(a,t){const e=Jn(t,this.size,2);a.uniform2fv(this.addr,e)}function Bu(a,t){const e=Jn(t,this.size,3);a.uniform3fv(this.addr,e)}function Gu(a,t){const e=Jn(t,this.size,4);a.uniform4fv(this.addr,e)}function Vu(a,t){const e=Jn(t,this.size,4);a.uniformMatrix2fv(this.addr,!1,e)}function ku(a,t){const e=Jn(t,this.size,9);a.uniformMatrix3fv(this.addr,!1,e)}function Hu(a,t){const e=Jn(t,this.size,16);a.uniformMatrix4fv(this.addr,!1,e)}function Wu(a,t){a.uniform1iv(this.addr,t)}function qu(a,t){a.uniform2iv(this.addr,t)}function Xu(a,t){a.uniform3iv(this.addr,t)}function Yu(a,t){a.uniform4iv(this.addr,t)}function ju(a,t){a.uniform1uiv(this.addr,t)}function Ju(a,t){a.uniform2uiv(this.addr,t)}function Zu(a,t){a.uniform3uiv(this.addr,t)}function Ku(a,t){a.uniform4uiv(this.addr,t)}function $u(a,t,e){const n=this.cache,i=t.length,r=Bi(e,i);$t(n,r)||(a.uniform1iv(this.addr,r),Qt(n,r));for(let o=0;o!==i;++o)e.setTexture2D(t[o]||Ha,r[o])}function Qu(a,t,e){const n=this.cache,i=t.length,r=Bi(e,i);$t(n,r)||(a.uniform1iv(this.addr,r),Qt(n,r));for(let o=0;o!==i;++o)e.setTexture3D(t[o]||qa,r[o])}function td(a,t,e){const n=this.cache,i=t.length,r=Bi(e,i);$t(n,r)||(a.uniform1iv(this.addr,r),Qt(n,r));for(let o=0;o!==i;++o)e.setTextureCube(t[o]||Xa,r[o])}function ed(a,t,e){const n=this.cache,i=t.length,r=Bi(e,i);$t(n,r)||(a.uniform1iv(this.addr,r),Qt(n,r));for(let o=0;o!==i;++o)e.setTexture2DArray(t[o]||Wa,r[o])}function nd(a){switch(a){case 5126:return Fu;case 35664:return zu;case 35665:return Bu;case 35666:return Gu;case 35674:return Vu;case 35675:return ku;case 35676:return Hu;case 5124:case 35670:return Wu;case 35667:case 35671:return qu;case 35668:case 35672:return Xu;case 35669:case 35673:return Yu;case 5125:return ju;case 36294:return Ju;case 36295:return Zu;case 36296:return Ku;case 35678:case 36198:case 36298:case 36306:case 35682:return $u;case 35679:case 36299:case 36307:return Qu;case 35680:case 36300:case 36308:case 36293:return td;case 36289:case 36303:case 36311:case 36292:return ed}}class id{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.setValue=Ou(e.type)}}class rd{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.size=e.size,this.setValue=nd(e.type)}}class sd{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,n){const i=this.seq;for(let r=0,o=i.length;r!==o;++r){const s=i[r];s.setValue(t,e[s.id],n)}}}const vr=/(\w+)(\])?(\[|\.)?/g;function ea(a,t){a.seq.push(t),a.map[t.id]=t}function ad(a,t,e){const n=a.name,i=n.length;for(vr.lastIndex=0;;){const r=vr.exec(n),o=vr.lastIndex;let s=r[1];const c=r[2]==="]",l=r[3];if(c&&(s=s|0),l===void 0||l==="["&&o+2===i){ea(e,l===void 0?new id(s,a,t):new rd(s,a,t));break}else{let d=e.map[s];d===void 0&&(d=new sd(s),ea(e,d)),e=d}}}class Ui{constructor(t,e){this.seq=[],this.map={};const n=t.getProgramParameter(e,35718);for(let i=0;i<n;++i){const r=t.getActiveUniform(e,i),o=t.getUniformLocation(e,r.name);ad(r,o,this)}}setValue(t,e,n,i){const r=this.map[e];r!==void 0&&r.setValue(t,n,i)}setOptional(t,e,n){const i=e[n];i!==void 0&&this.setValue(t,n,i)}static upload(t,e,n,i){for(let r=0,o=e.length;r!==o;++r){const s=e[r],c=n[s.id];c.needsUpdate!==!1&&s.setValue(t,c.value,i)}}static seqWithValue(t,e){const n=[];for(let i=0,r=t.length;i!==r;++i){const o=t[i];o.id in e&&n.push(o)}return n}}function na(a,t,e){const n=a.createShader(t);return a.shaderSource(n,e),a.compileShader(n),n}let od=0;function ld(a,t){const e=a.split(`
`),n=[],i=Math.max(t-6,0),r=Math.min(t+6,e.length);for(let o=i;o<r;o++){const s=o+1;n.push(`${s===t?">":" "} ${s}: ${e[o]}`)}return n.join(`
`)}function cd(a){switch(a){case gn:return["Linear","( value )"];case Gt:return["sRGB","( value )"];default:return console.warn("THREE.WebGLProgram: Unsupported encoding:",a),["Linear","( value )"]}}function ia(a,t,e){const n=a.getShaderParameter(t,35713),i=a.getShaderInfoLog(t).trim();if(n&&i==="")return"";const r=/ERROR: 0:(\d+)/.exec(i);if(r){const o=parseInt(r[1]);return e.toUpperCase()+`

`+i+`

`+ld(a.getShaderSource(t),o)}else return i}function hd(a,t){const e=cd(t);return"vec4 "+a+"( vec4 value ) { return LinearTo"+e[0]+e[1]+"; }"}function ud(a,t){let e;switch(t){case To:e="Linear";break;case Ao:e="Reinhard";break;case Co:e="OptimizedCineon";break;case wa:e="ACESFilmic";break;case Lo:e="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",t),e="Linear"}return"vec3 "+a+"( vec3 color ) { return "+e+"ToneMapping( color ); }"}function dd(a){return[a.extensionDerivatives||a.envMapCubeUVHeight||a.bumpMap||a.normalMapTangentSpace||a.clearcoatNormalMap||a.flatShading||a.shaderID==="physical"?"#extension GL_OES_standard_derivatives : enable":"",(a.extensionFragDepth||a.logarithmicDepthBuffer)&&a.rendererExtensionFragDepth?"#extension GL_EXT_frag_depth : enable":"",a.extensionDrawBuffers&&a.rendererExtensionDrawBuffers?"#extension GL_EXT_draw_buffers : require":"",(a.extensionShaderTextureLOD||a.envMap||a.transmission)&&a.rendererExtensionShaderTextureLod?"#extension GL_EXT_shader_texture_lod : enable":""].filter(ni).join(`
`)}function fd(a){const t=[];for(const e in a){const n=a[e];n!==!1&&t.push("#define "+e+" "+n)}return t.join(`
`)}function pd(a,t){const e={},n=a.getProgramParameter(t,35721);for(let i=0;i<n;i++){const r=a.getActiveAttrib(t,i),o=r.name;let s=1;r.type===35674&&(s=2),r.type===35675&&(s=3),r.type===35676&&(s=4),e[o]={type:r.type,location:a.getAttribLocation(t,o),locationSize:s}}return e}function ni(a){return a!==""}function ra(a,t){const e=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return a.replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,e).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows)}function sa(a,t){return a.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}const md=/^[ \t]*#include +<([\w\d./]+)>/gm;function Ir(a){return a.replace(md,gd)}function gd(a,t){const e=Ct[t];if(e===void 0)throw new Error("Can not resolve #include <"+t+">");return Ir(e)}const _d=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function aa(a){return a.replace(_d,vd)}function vd(a,t,e,n){let i="";for(let r=parseInt(t);r<parseInt(e);r++)i+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return i}function oa(a){let t="precision "+a.precision+` float;
precision `+a.precision+" int;";return a.precision==="highp"?t+=`
#define HIGH_PRECISION`:a.precision==="mediump"?t+=`
#define MEDIUM_PRECISION`:a.precision==="lowp"&&(t+=`
#define LOW_PRECISION`),t}function xd(a){let t="SHADOWMAP_TYPE_BASIC";return a.shadowMapType===xa?t="SHADOWMAP_TYPE_PCF":a.shadowMapType===Ma?t="SHADOWMAP_TYPE_PCF_SOFT":a.shadowMapType===ei&&(t="SHADOWMAP_TYPE_VSM"),t}function Md(a){let t="ENVMAP_TYPE_CUBE";if(a.envMap)switch(a.envMapMode){case Wn:case qn:t="ENVMAP_TYPE_CUBE";break;case Oi:t="ENVMAP_TYPE_CUBE_UV";break}return t}function Sd(a){let t="ENVMAP_MODE_REFLECTION";if(a.envMap)switch(a.envMapMode){case qn:t="ENVMAP_MODE_REFRACTION";break}return t}function yd(a){let t="ENVMAP_BLENDING_NONE";if(a.envMap)switch(a.combine){case ba:t="ENVMAP_BLENDING_MULTIPLY";break;case wo:t="ENVMAP_BLENDING_MIX";break;case Eo:t="ENVMAP_BLENDING_ADD";break}return t}function bd(a){const t=a.envMapCubeUVHeight;if(t===null)return null;const e=Math.log2(t)-2,n=1/t;return{texelWidth:1/(3*Math.max(Math.pow(2,e),112)),texelHeight:n,maxMip:e}}function wd(a,t,e,n){const i=a.getContext(),r=e.defines;let o=e.vertexShader,s=e.fragmentShader;const c=xd(e),l=Md(e),h=Sd(e),d=yd(e),f=bd(e),p=e.isWebGL2?"":dd(e),g=fd(r),_=i.createProgram();let m,u,E=e.glslVersion?"#version "+e.glslVersion+`
`:"";e.isRawShaderMaterial?(m=[g].filter(ni).join(`
`),m.length>0&&(m+=`
`),u=[p,g].filter(ni).join(`
`),u.length>0&&(u+=`
`)):(m=[oa(e),"#define SHADER_NAME "+e.shaderName,g,e.instancing?"#define USE_INSTANCING":"",e.instancingColor?"#define USE_INSTANCING_COLOR":"",e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+h:"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.displacementMap?"#define USE_DISPLACEMENTMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.mapUv?"#define MAP_UV "+e.mapUv:"",e.alphaMapUv?"#define ALPHAMAP_UV "+e.alphaMapUv:"",e.lightMapUv?"#define LIGHTMAP_UV "+e.lightMapUv:"",e.aoMapUv?"#define AOMAP_UV "+e.aoMapUv:"",e.emissiveMapUv?"#define EMISSIVEMAP_UV "+e.emissiveMapUv:"",e.bumpMapUv?"#define BUMPMAP_UV "+e.bumpMapUv:"",e.normalMapUv?"#define NORMALMAP_UV "+e.normalMapUv:"",e.displacementMapUv?"#define DISPLACEMENTMAP_UV "+e.displacementMapUv:"",e.metalnessMapUv?"#define METALNESSMAP_UV "+e.metalnessMapUv:"",e.roughnessMapUv?"#define ROUGHNESSMAP_UV "+e.roughnessMapUv:"",e.clearcoatMapUv?"#define CLEARCOATMAP_UV "+e.clearcoatMapUv:"",e.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+e.clearcoatNormalMapUv:"",e.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+e.clearcoatRoughnessMapUv:"",e.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+e.iridescenceMapUv:"",e.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+e.iridescenceThicknessMapUv:"",e.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+e.sheenColorMapUv:"",e.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+e.sheenRoughnessMapUv:"",e.specularMapUv?"#define SPECULARMAP_UV "+e.specularMapUv:"",e.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+e.specularColorMapUv:"",e.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+e.specularIntensityMapUv:"",e.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+e.transmissionMapUv:"",e.thicknessMapUv?"#define THICKNESSMAP_UV "+e.thicknessMapUv:"",e.vertexTangents?"#define USE_TANGENT":"",e.vertexColors?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUvs2?"#define USE_UV2":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.flatShading?"#define FLAT_SHADED":"",e.skinning?"#define USE_SKINNING":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals&&e.flatShading===!1?"#define USE_MORPHNORMALS":"",e.morphColors&&e.isWebGL2?"#define USE_MORPHCOLORS":"",e.morphTargetsCount>0&&e.isWebGL2?"#define MORPHTARGETS_TEXTURE":"",e.morphTargetsCount>0&&e.isWebGL2?"#define MORPHTARGETS_TEXTURE_STRIDE "+e.morphTextureStride:"",e.morphTargetsCount>0&&e.isWebGL2?"#define MORPHTARGETS_COUNT "+e.morphTargetsCount:"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.logarithmicDepthBuffer&&e.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#if ( defined( USE_MORPHTARGETS ) && ! defined( MORPHTARGETS_TEXTURE ) )","	attribute vec3 morphTarget0;","	attribute vec3 morphTarget1;","	attribute vec3 morphTarget2;","	attribute vec3 morphTarget3;","	#ifdef USE_MORPHNORMALS","		attribute vec3 morphNormal0;","		attribute vec3 morphNormal1;","		attribute vec3 morphNormal2;","		attribute vec3 morphNormal3;","	#else","		attribute vec3 morphTarget4;","		attribute vec3 morphTarget5;","		attribute vec3 morphTarget6;","		attribute vec3 morphTarget7;","	#endif","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(ni).join(`
`),u=[p,oa(e),"#define SHADER_NAME "+e.shaderName,g,e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.matcap?"#define USE_MATCAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+l:"",e.envMap?"#define "+h:"",e.envMap?"#define "+d:"",f?"#define CUBEUV_TEXEL_WIDTH "+f.texelWidth:"",f?"#define CUBEUV_TEXEL_HEIGHT "+f.texelHeight:"",f?"#define CUBEUV_MAX_MIP "+f.maxMip+".0":"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.clearcoat?"#define USE_CLEARCOAT":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.iridescence?"#define USE_IRIDESCENCE":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaTest?"#define USE_ALPHATEST":"",e.sheen?"#define USE_SHEEN":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",e.vertexTangents?"#define USE_TANGENT":"",e.vertexColors||e.instancingColor?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUvs2?"#define USE_UV2":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.gradientMap?"#define USE_GRADIENTMAP":"",e.flatShading?"#define FLAT_SHADED":"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",e.useLegacyLights?"#define LEGACY_LIGHTS":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.logarithmicDepthBuffer&&e.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",e.toneMapping!==je?"#define TONE_MAPPING":"",e.toneMapping!==je?Ct.tonemapping_pars_fragment:"",e.toneMapping!==je?ud("toneMapping",e.toneMapping):"",e.dithering?"#define DITHERING":"",e.opaque?"#define OPAQUE":"",Ct.encodings_pars_fragment,hd("linearToOutputTexel",e.outputEncoding),e.useDepthPacking?"#define DEPTH_PACKING "+e.depthPacking:"",`
`].filter(ni).join(`
`)),o=Ir(o),o=ra(o,e),o=sa(o,e),s=Ir(s),s=ra(s,e),s=sa(s,e),o=aa(o),s=aa(s),e.isWebGL2&&e.isRawShaderMaterial!==!0&&(E=`#version 300 es
`,m=["precision mediump sampler2DArray;","#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,u=["#define varying in",e.glslVersion===As?"":"layout(location = 0) out highp vec4 pc_fragColor;",e.glslVersion===As?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+u);const x=E+m+o,b=E+u+s,y=na(i,35633,x),C=na(i,35632,b);if(i.attachShader(_,y),i.attachShader(_,C),e.index0AttributeName!==void 0?i.bindAttribLocation(_,0,e.index0AttributeName):e.morphTargets===!0&&i.bindAttribLocation(_,0,"position"),i.linkProgram(_),a.debug.checkShaderErrors){const M=i.getProgramInfoLog(_).trim(),T=i.getShaderInfoLog(y).trim(),X=i.getShaderInfoLog(C).trim();let q=!0,I=!0;if(i.getProgramParameter(_,35714)===!1)if(q=!1,typeof a.debug.onShaderError=="function")a.debug.onShaderError(i,_,y,C);else{const N=ia(i,y,"vertex"),B=ia(i,C,"fragment");console.error("THREE.WebGLProgram: Shader Error "+i.getError()+" - VALIDATE_STATUS "+i.getProgramParameter(_,35715)+`

Program Info Log: `+M+`
`+N+`
`+B)}else M!==""?console.warn("THREE.WebGLProgram: Program Info Log:",M):(T===""||X==="")&&(I=!1);I&&(this.diagnostics={runnable:q,programLog:M,vertexShader:{log:T,prefix:m},fragmentShader:{log:X,prefix:u}})}i.deleteShader(y),i.deleteShader(C);let D;this.getUniforms=function(){return D===void 0&&(D=new Ui(i,_)),D};let R;return this.getAttributes=function(){return R===void 0&&(R=pd(i,_)),R},this.destroy=function(){n.releaseStatesOfProgram(this),i.deleteProgram(_),this.program=void 0},this.name=e.shaderName,this.id=od++,this.cacheKey=t,this.usedTimes=1,this.program=_,this.vertexShader=y,this.fragmentShader=C,this}let Ed=0;class Td{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t){const e=t.vertexShader,n=t.fragmentShader,i=this._getShaderStage(e),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(t);return o.has(i)===!1&&(o.add(i),i.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(t){const e=this.materialCache.get(t);for(const n of e)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(t),this}getVertexShaderID(t){return this._getShaderStage(t.vertexShader).id}getFragmentShaderID(t){return this._getShaderStage(t.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){const e=this.materialCache;let n=e.get(t);return n===void 0&&(n=new Set,e.set(t,n)),n}_getShaderStage(t){const e=this.shaderCache;let n=e.get(t);return n===void 0&&(n=new Ad(t),e.set(t,n)),n}}class Ad{constructor(t){this.id=Ed++,this.code=t,this.usedTimes=0}}function Cd(a,t,e,n,i,r,o){const s=new Or,c=new Td,l=[],h=i.isWebGL2,d=i.logarithmicDepthBuffer,f=i.vertexTextures;let p=i.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(M){return M===1?"uv2":"uv"}function m(M,T,X,q,I){const N=q.fog,B=I.geometry,Q=M.isMeshStandardMaterial?q.environment:null,et=(M.isMeshStandardMaterial?e:t).get(M.envMap||Q),j=et&&et.mapping===Oi?et.image.height:null,Z=g[M.type];M.precision!==null&&(p=i.getMaxPrecision(M.precision),p!==M.precision&&console.warn("THREE.WebGLProgram.getParameters:",M.precision,"not supported, using",p,"instead."));const nt=B.morphAttributes.position||B.morphAttributes.normal||B.morphAttributes.color,vt=nt!==void 0?nt.length:0;let ct=0;B.morphAttributes.position!==void 0&&(ct=1),B.morphAttributes.normal!==void 0&&(ct=2),B.morphAttributes.color!==void 0&&(ct=3);let V,J,st,lt;if(Z){const ut=Fe[Z];V=ut.vertexShader,J=ut.fragmentShader}else V=M.vertexShader,J=M.fragmentShader,c.update(M),st=c.getVertexShaderID(M),lt=c.getFragmentShaderID(M);const z=a.getRenderTarget(),bt=I.isInstancedMesh===!0,St=!!M.map,it=!!M.matcap,Mt=!!et,Ft=!!M.aoMap,gt=!!M.lightMap,Tt=!!M.bumpMap,Wt=!!M.normalMap,kt=!!M.displacementMap,jt=!!M.emissiveMap,qt=!!M.metalnessMap,Nt=!!M.roughnessMap,Bt=M.clearcoat>0,ae=M.iridescence>0,w=M.sheen>0,v=M.transmission>0,G=Bt&&!!M.clearcoatMap,K=Bt&&!!M.clearcoatNormalMap,tt=Bt&&!!M.clearcoatRoughnessMap,at=ae&&!!M.iridescenceMap,A=ae&&!!M.iridescenceThicknessMap,Y=w&&!!M.sheenColorMap,F=w&&!!M.sheenRoughnessMap,ot=!!M.specularMap,ft=!!M.specularColorMap,mt=!!M.specularIntensityMap,ht=v&&!!M.transmissionMap,pt=v&&!!M.thicknessMap,wt=!!M.gradientMap,Lt=!!M.alphaMap,Xt=M.alphaTest>0,L=!!M.extensions,H=!!B.attributes.uv2;return{isWebGL2:h,shaderID:Z,shaderName:M.type,vertexShader:V,fragmentShader:J,defines:M.defines,customVertexShaderID:st,customFragmentShaderID:lt,isRawShaderMaterial:M.isRawShaderMaterial===!0,glslVersion:M.glslVersion,precision:p,instancing:bt,instancingColor:bt&&I.instanceColor!==null,supportsVertexTextures:f,outputEncoding:z===null?a.outputEncoding:z.isXRRenderTarget===!0?z.texture.encoding:gn,map:St,matcap:it,envMap:Mt,envMapMode:Mt&&et.mapping,envMapCubeUVHeight:j,aoMap:Ft,lightMap:gt,bumpMap:Tt,normalMap:Wt,displacementMap:f&&kt,emissiveMap:jt,normalMapObjectSpace:Wt&&M.normalMapType===jo,normalMapTangentSpace:Wt&&M.normalMapType===Aa,decodeVideoTexture:St&&M.map.isVideoTexture===!0&&M.map.encoding===Gt,metalnessMap:qt,roughnessMap:Nt,clearcoat:Bt,clearcoatMap:G,clearcoatNormalMap:K,clearcoatRoughnessMap:tt,iridescence:ae,iridescenceMap:at,iridescenceThicknessMap:A,sheen:w,sheenColorMap:Y,sheenRoughnessMap:F,specularMap:ot,specularColorMap:ft,specularIntensityMap:mt,transmission:v,transmissionMap:ht,thicknessMap:pt,gradientMap:wt,opaque:M.transparent===!1&&M.blending===Vn,alphaMap:Lt,alphaTest:Xt,combine:M.combine,mapUv:St&&_(M.map.channel),aoMapUv:Ft&&_(M.aoMap.channel),lightMapUv:gt&&_(M.lightMap.channel),bumpMapUv:Tt&&_(M.bumpMap.channel),normalMapUv:Wt&&_(M.normalMap.channel),displacementMapUv:kt&&_(M.displacementMap.channel),emissiveMapUv:jt&&_(M.emissiveMap.channel),metalnessMapUv:qt&&_(M.metalnessMap.channel),roughnessMapUv:Nt&&_(M.roughnessMap.channel),clearcoatMapUv:G&&_(M.clearcoatMap.channel),clearcoatNormalMapUv:K&&_(M.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:tt&&_(M.clearcoatRoughnessMap.channel),iridescenceMapUv:at&&_(M.iridescenceMap.channel),iridescenceThicknessMapUv:A&&_(M.iridescenceThicknessMap.channel),sheenColorMapUv:Y&&_(M.sheenColorMap.channel),sheenRoughnessMapUv:F&&_(M.sheenRoughnessMap.channel),specularMapUv:ot&&_(M.specularMap.channel),specularColorMapUv:ft&&_(M.specularColorMap.channel),specularIntensityMapUv:mt&&_(M.specularIntensityMap.channel),transmissionMapUv:ht&&_(M.transmissionMap.channel),thicknessMapUv:pt&&_(M.thicknessMap.channel),alphaMapUv:Lt&&_(M.alphaMap.channel),vertexTangents:Wt&&!!B.attributes.tangent,vertexColors:M.vertexColors,vertexAlphas:M.vertexColors===!0&&!!B.attributes.color&&B.attributes.color.itemSize===4,vertexUvs2:H,pointsUvs:I.isPoints===!0&&!!B.attributes.uv&&(St||Lt),fog:!!N,useFog:M.fog===!0,fogExp2:N&&N.isFogExp2,flatShading:M.flatShading===!0,sizeAttenuation:M.sizeAttenuation===!0,logarithmicDepthBuffer:d,skinning:I.isSkinnedMesh===!0,morphTargets:B.morphAttributes.position!==void 0,morphNormals:B.morphAttributes.normal!==void 0,morphColors:B.morphAttributes.color!==void 0,morphTargetsCount:vt,morphTextureStride:ct,numDirLights:T.directional.length,numPointLights:T.point.length,numSpotLights:T.spot.length,numSpotLightMaps:T.spotLightMap.length,numRectAreaLights:T.rectArea.length,numHemiLights:T.hemi.length,numDirLightShadows:T.directionalShadowMap.length,numPointLightShadows:T.pointShadowMap.length,numSpotLightShadows:T.spotShadowMap.length,numSpotLightShadowsWithMaps:T.numSpotLightShadowsWithMaps,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:M.dithering,shadowMapEnabled:a.shadowMap.enabled&&X.length>0,shadowMapType:a.shadowMap.type,toneMapping:M.toneMapped?a.toneMapping:je,useLegacyLights:a.useLegacyLights,premultipliedAlpha:M.premultipliedAlpha,doubleSided:M.side===Xe,flipSided:M.side===pe,useDepthPacking:M.depthPacking>=0,depthPacking:M.depthPacking||0,index0AttributeName:M.index0AttributeName,extensionDerivatives:L&&M.extensions.derivatives===!0,extensionFragDepth:L&&M.extensions.fragDepth===!0,extensionDrawBuffers:L&&M.extensions.drawBuffers===!0,extensionShaderTextureLOD:L&&M.extensions.shaderTextureLOD===!0,rendererExtensionFragDepth:h||n.has("EXT_frag_depth"),rendererExtensionDrawBuffers:h||n.has("WEBGL_draw_buffers"),rendererExtensionShaderTextureLod:h||n.has("EXT_shader_texture_lod"),customProgramCacheKey:M.customProgramCacheKey()}}function u(M){const T=[];if(M.shaderID?T.push(M.shaderID):(T.push(M.customVertexShaderID),T.push(M.customFragmentShaderID)),M.defines!==void 0)for(const X in M.defines)T.push(X),T.push(M.defines[X]);return M.isRawShaderMaterial===!1&&(E(T,M),x(T,M),T.push(a.outputEncoding)),T.push(M.customProgramCacheKey),T.join()}function E(M,T){M.push(T.precision),M.push(T.outputEncoding),M.push(T.envMapMode),M.push(T.envMapCubeUVHeight),M.push(T.mapUv),M.push(T.alphaMapUv),M.push(T.lightMapUv),M.push(T.aoMapUv),M.push(T.bumpMapUv),M.push(T.normalMapUv),M.push(T.displacementMapUv),M.push(T.emissiveMapUv),M.push(T.metalnessMapUv),M.push(T.roughnessMapUv),M.push(T.clearcoatMapUv),M.push(T.clearcoatNormalMapUv),M.push(T.clearcoatRoughnessMapUv),M.push(T.iridescenceMapUv),M.push(T.iridescenceThicknessMapUv),M.push(T.sheenColorMapUv),M.push(T.sheenRoughnessMapUv),M.push(T.specularMapUv),M.push(T.specularColorMapUv),M.push(T.specularIntensityMapUv),M.push(T.transmissionMapUv),M.push(T.thicknessMapUv),M.push(T.combine),M.push(T.fogExp2),M.push(T.sizeAttenuation),M.push(T.morphTargetsCount),M.push(T.morphAttributeCount),M.push(T.numDirLights),M.push(T.numPointLights),M.push(T.numSpotLights),M.push(T.numSpotLightMaps),M.push(T.numHemiLights),M.push(T.numRectAreaLights),M.push(T.numDirLightShadows),M.push(T.numPointLightShadows),M.push(T.numSpotLightShadows),M.push(T.numSpotLightShadowsWithMaps),M.push(T.shadowMapType),M.push(T.toneMapping),M.push(T.numClippingPlanes),M.push(T.numClipIntersection),M.push(T.depthPacking)}function x(M,T){s.disableAll(),T.isWebGL2&&s.enable(0),T.supportsVertexTextures&&s.enable(1),T.instancing&&s.enable(2),T.instancingColor&&s.enable(3),T.matcap&&s.enable(4),T.envMap&&s.enable(5),T.normalMapObjectSpace&&s.enable(6),T.normalMapTangentSpace&&s.enable(7),T.clearcoat&&s.enable(8),T.iridescence&&s.enable(9),T.alphaTest&&s.enable(10),T.vertexColors&&s.enable(11),T.vertexAlphas&&s.enable(12),T.vertexUvs2&&s.enable(13),T.vertexTangents&&s.enable(14),M.push(s.mask),s.disableAll(),T.fog&&s.enable(0),T.useFog&&s.enable(1),T.flatShading&&s.enable(2),T.logarithmicDepthBuffer&&s.enable(3),T.skinning&&s.enable(4),T.morphTargets&&s.enable(5),T.morphNormals&&s.enable(6),T.morphColors&&s.enable(7),T.premultipliedAlpha&&s.enable(8),T.shadowMapEnabled&&s.enable(9),T.useLegacyLights&&s.enable(10),T.doubleSided&&s.enable(11),T.flipSided&&s.enable(12),T.useDepthPacking&&s.enable(13),T.dithering&&s.enable(14),T.transmission&&s.enable(15),T.sheen&&s.enable(16),T.decodeVideoTexture&&s.enable(17),T.opaque&&s.enable(18),T.pointsUvs&&s.enable(19),M.push(s.mask)}function b(M){const T=g[M.type];let X;if(T){const q=Fe[T];X=_l.clone(q.uniforms)}else X=M.uniforms;return X}function y(M,T){let X;for(let q=0,I=l.length;q<I;q++){const N=l[q];if(N.cacheKey===T){X=N,++X.usedTimes;break}}return X===void 0&&(X=new wd(a,T,M,r),l.push(X)),X}function C(M){if(--M.usedTimes===0){const T=l.indexOf(M);l[T]=l[l.length-1],l.pop(),M.destroy()}}function D(M){c.remove(M)}function R(){c.dispose()}return{getParameters:m,getProgramCacheKey:u,getUniforms:b,acquireProgram:y,releaseProgram:C,releaseShaderCache:D,programs:l,dispose:R}}function Ld(){let a=new WeakMap;function t(r){let o=a.get(r);return o===void 0&&(o={},a.set(r,o)),o}function e(r){a.delete(r)}function n(r,o,s){a.get(r)[o]=s}function i(){a=new WeakMap}return{get:t,remove:e,update:n,dispose:i}}function Pd(a,t){return a.groupOrder!==t.groupOrder?a.groupOrder-t.groupOrder:a.renderOrder!==t.renderOrder?a.renderOrder-t.renderOrder:a.material.id!==t.material.id?a.material.id-t.material.id:a.z!==t.z?a.z-t.z:a.id-t.id}function la(a,t){return a.groupOrder!==t.groupOrder?a.groupOrder-t.groupOrder:a.renderOrder!==t.renderOrder?a.renderOrder-t.renderOrder:a.z!==t.z?t.z-a.z:a.id-t.id}function ca(){const a=[];let t=0;const e=[],n=[],i=[];function r(){t=0,e.length=0,n.length=0,i.length=0}function o(d,f,p,g,_,m){let u=a[t];return u===void 0?(u={id:d.id,object:d,geometry:f,material:p,groupOrder:g,renderOrder:d.renderOrder,z:_,group:m},a[t]=u):(u.id=d.id,u.object=d,u.geometry=f,u.material=p,u.groupOrder=g,u.renderOrder=d.renderOrder,u.z=_,u.group=m),t++,u}function s(d,f,p,g,_,m){const u=o(d,f,p,g,_,m);p.transmission>0?n.push(u):p.transparent===!0?i.push(u):e.push(u)}function c(d,f,p,g,_,m){const u=o(d,f,p,g,_,m);p.transmission>0?n.unshift(u):p.transparent===!0?i.unshift(u):e.unshift(u)}function l(d,f){e.length>1&&e.sort(d||Pd),n.length>1&&n.sort(f||la),i.length>1&&i.sort(f||la)}function h(){for(let d=t,f=a.length;d<f;d++){const p=a[d];if(p.id===null)break;p.id=null,p.object=null,p.geometry=null,p.material=null,p.group=null}}return{opaque:e,transmissive:n,transparent:i,init:r,push:s,unshift:c,finish:h,sort:l}}function Rd(){let a=new WeakMap;function t(n,i){const r=a.get(n);let o;return r===void 0?(o=new ca,a.set(n,[o])):i>=r.length?(o=new ca,r.push(o)):o=r[i],o}function e(){a=new WeakMap}return{get:t,dispose:e}}function Dd(){const a={};return{get:function(t){if(a[t.id]!==void 0)return a[t.id];let e;switch(t.type){case"DirectionalLight":e={direction:new P,color:new Ot};break;case"SpotLight":e={position:new P,direction:new P,color:new Ot,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":e={position:new P,color:new Ot,distance:0,decay:0};break;case"HemisphereLight":e={direction:new P,skyColor:new Ot,groundColor:new Ot};break;case"RectAreaLight":e={color:new Ot,position:new P,halfWidth:new P,halfHeight:new P};break}return a[t.id]=e,e}}}function Id(){const a={};return{get:function(t){if(a[t.id]!==void 0)return a[t.id];let e;switch(t.type){case"DirectionalLight":e={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new dt};break;case"SpotLight":e={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new dt};break;case"PointLight":e={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new dt,shadowCameraNear:1,shadowCameraFar:1e3};break}return a[t.id]=e,e}}}let Ud=0;function Nd(a,t){return(t.castShadow?2:0)-(a.castShadow?2:0)+(t.map?1:0)-(a.map?1:0)}function Od(a,t){const e=new Dd,n=Id(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0};for(let h=0;h<9;h++)i.probe.push(new P);const r=new P,o=new Kt,s=new Kt;function c(h,d){let f=0,p=0,g=0;for(let X=0;X<9;X++)i.probe[X].set(0,0,0);let _=0,m=0,u=0,E=0,x=0,b=0,y=0,C=0,D=0,R=0;h.sort(Nd);const M=d===!0?Math.PI:1;for(let X=0,q=h.length;X<q;X++){const I=h[X],N=I.color,B=I.intensity,Q=I.distance,et=I.shadow&&I.shadow.map?I.shadow.map.texture:null;if(I.isAmbientLight)f+=N.r*B*M,p+=N.g*B*M,g+=N.b*B*M;else if(I.isLightProbe)for(let j=0;j<9;j++)i.probe[j].addScaledVector(I.sh.coefficients[j],B);else if(I.isDirectionalLight){const j=e.get(I);if(j.color.copy(I.color).multiplyScalar(I.intensity*M),I.castShadow){const Z=I.shadow,nt=n.get(I);nt.shadowBias=Z.bias,nt.shadowNormalBias=Z.normalBias,nt.shadowRadius=Z.radius,nt.shadowMapSize=Z.mapSize,i.directionalShadow[_]=nt,i.directionalShadowMap[_]=et,i.directionalShadowMatrix[_]=I.shadow.matrix,b++}i.directional[_]=j,_++}else if(I.isSpotLight){const j=e.get(I);j.position.setFromMatrixPosition(I.matrixWorld),j.color.copy(N).multiplyScalar(B*M),j.distance=Q,j.coneCos=Math.cos(I.angle),j.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),j.decay=I.decay,i.spot[u]=j;const Z=I.shadow;if(I.map&&(i.spotLightMap[D]=I.map,D++,Z.updateMatrices(I),I.castShadow&&R++),i.spotLightMatrix[u]=Z.matrix,I.castShadow){const nt=n.get(I);nt.shadowBias=Z.bias,nt.shadowNormalBias=Z.normalBias,nt.shadowRadius=Z.radius,nt.shadowMapSize=Z.mapSize,i.spotShadow[u]=nt,i.spotShadowMap[u]=et,C++}u++}else if(I.isRectAreaLight){const j=e.get(I);j.color.copy(N).multiplyScalar(B),j.halfWidth.set(I.width*.5,0,0),j.halfHeight.set(0,I.height*.5,0),i.rectArea[E]=j,E++}else if(I.isPointLight){const j=e.get(I);if(j.color.copy(I.color).multiplyScalar(I.intensity*M),j.distance=I.distance,j.decay=I.decay,I.castShadow){const Z=I.shadow,nt=n.get(I);nt.shadowBias=Z.bias,nt.shadowNormalBias=Z.normalBias,nt.shadowRadius=Z.radius,nt.shadowMapSize=Z.mapSize,nt.shadowCameraNear=Z.camera.near,nt.shadowCameraFar=Z.camera.far,i.pointShadow[m]=nt,i.pointShadowMap[m]=et,i.pointShadowMatrix[m]=I.shadow.matrix,y++}i.point[m]=j,m++}else if(I.isHemisphereLight){const j=e.get(I);j.skyColor.copy(I.color).multiplyScalar(B*M),j.groundColor.copy(I.groundColor).multiplyScalar(B*M),i.hemi[x]=j,x++}}E>0&&(t.isWebGL2||a.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=rt.LTC_FLOAT_1,i.rectAreaLTC2=rt.LTC_FLOAT_2):a.has("OES_texture_half_float_linear")===!0?(i.rectAreaLTC1=rt.LTC_HALF_1,i.rectAreaLTC2=rt.LTC_HALF_2):console.error("THREE.WebGLRenderer: Unable to use RectAreaLight. Missing WebGL extensions.")),i.ambient[0]=f,i.ambient[1]=p,i.ambient[2]=g;const T=i.hash;(T.directionalLength!==_||T.pointLength!==m||T.spotLength!==u||T.rectAreaLength!==E||T.hemiLength!==x||T.numDirectionalShadows!==b||T.numPointShadows!==y||T.numSpotShadows!==C||T.numSpotMaps!==D)&&(i.directional.length=_,i.spot.length=u,i.rectArea.length=E,i.point.length=m,i.hemi.length=x,i.directionalShadow.length=b,i.directionalShadowMap.length=b,i.pointShadow.length=y,i.pointShadowMap.length=y,i.spotShadow.length=C,i.spotShadowMap.length=C,i.directionalShadowMatrix.length=b,i.pointShadowMatrix.length=y,i.spotLightMatrix.length=C+D-R,i.spotLightMap.length=D,i.numSpotLightShadowsWithMaps=R,T.directionalLength=_,T.pointLength=m,T.spotLength=u,T.rectAreaLength=E,T.hemiLength=x,T.numDirectionalShadows=b,T.numPointShadows=y,T.numSpotShadows=C,T.numSpotMaps=D,i.version=Ud++)}function l(h,d){let f=0,p=0,g=0,_=0,m=0;const u=d.matrixWorldInverse;for(let E=0,x=h.length;E<x;E++){const b=h[E];if(b.isDirectionalLight){const y=i.directional[f];y.direction.setFromMatrixPosition(b.matrixWorld),r.setFromMatrixPosition(b.target.matrixWorld),y.direction.sub(r),y.direction.transformDirection(u),f++}else if(b.isSpotLight){const y=i.spot[g];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(u),y.direction.setFromMatrixPosition(b.matrixWorld),r.setFromMatrixPosition(b.target.matrixWorld),y.direction.sub(r),y.direction.transformDirection(u),g++}else if(b.isRectAreaLight){const y=i.rectArea[_];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(u),s.identity(),o.copy(b.matrixWorld),o.premultiply(u),s.extractRotation(o),y.halfWidth.set(b.width*.5,0,0),y.halfHeight.set(0,b.height*.5,0),y.halfWidth.applyMatrix4(s),y.halfHeight.applyMatrix4(s),_++}else if(b.isPointLight){const y=i.point[p];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(u),p++}else if(b.isHemisphereLight){const y=i.hemi[m];y.direction.setFromMatrixPosition(b.matrixWorld),y.direction.transformDirection(u),m++}}}return{setup:c,setupView:l,state:i}}function ha(a,t){const e=new Od(a,t),n=[],i=[];function r(){n.length=0,i.length=0}function o(d){n.push(d)}function s(d){i.push(d)}function c(d){e.setup(n,d)}function l(d){e.setupView(n,d)}return{init:r,state:{lightsArray:n,shadowsArray:i,lights:e},setupLights:c,setupLightsView:l,pushLight:o,pushShadow:s}}function Fd(a,t){let e=new WeakMap;function n(r,o=0){const s=e.get(r);let c;return s===void 0?(c=new ha(a,t),e.set(r,[c])):o>=s.length?(c=new ha(a,t),s.push(c)):c=s[o],c}function i(){e=new WeakMap}return{get:n,dispose:i}}class zd extends hi{constructor(t){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Xo,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}}class Bd extends hi{constructor(t){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}}const Gd=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,Vd=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function kd(a,t,e){let n=new Fr;const i=new dt,r=new dt,o=new ie,s=new zd({depthPacking:Yo}),c=new Bd,l={},h=e.maxTextureSize,d={[en]:pe,[pe]:en,[Xe]:Xe},f=new xn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new dt},radius:{value:4}},vertexShader:Gd,fragmentShader:Vd}),p=f.clone();p.defines.HORIZONTAL_PASS=1;const g=new Ee;g.setAttribute("position",new Ie(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const _=new ze(g,f),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=xa,this.render=function(b,y,C){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||b.length===0)return;const D=a.getRenderTarget(),R=a.getActiveCubeFace(),M=a.getActiveMipmapLevel(),T=a.state;T.setBlending(tn),T.buffers.color.setClear(1,1,1,1),T.buffers.depth.setTest(!0),T.setScissorTest(!1);for(let X=0,q=b.length;X<q;X++){const I=b[X],N=I.shadow;if(N===void 0){console.warn("THREE.WebGLShadowMap:",I,"has no shadow.");continue}if(N.autoUpdate===!1&&N.needsUpdate===!1)continue;i.copy(N.mapSize);const B=N.getFrameExtents();if(i.multiply(B),r.copy(N.mapSize),(i.x>h||i.y>h)&&(i.x>h&&(r.x=Math.floor(h/B.x),i.x=r.x*B.x,N.mapSize.x=r.x),i.y>h&&(r.y=Math.floor(h/B.y),i.y=r.y*B.y,N.mapSize.y=r.y)),N.map===null){const et=this.type!==ei?{minFilter:fe,magFilter:fe}:{};N.map=new _n(i.x,i.y,et),N.map.texture.name=I.name+".shadowMap",N.camera.updateProjectionMatrix()}a.setRenderTarget(N.map),a.clear();const Q=N.getViewportCount();for(let et=0;et<Q;et++){const j=N.getViewport(et);o.set(r.x*j.x,r.y*j.y,r.x*j.z,r.y*j.w),T.viewport(o),N.updateMatrices(I,et),n=N.getFrustum(),x(y,C,N.camera,I,this.type)}N.isPointLightShadow!==!0&&this.type===ei&&u(N,C),N.needsUpdate=!1}m.needsUpdate=!1,a.setRenderTarget(D,R,M)};function u(b,y){const C=t.update(_);f.defines.VSM_SAMPLES!==b.blurSamples&&(f.defines.VSM_SAMPLES=b.blurSamples,p.defines.VSM_SAMPLES=b.blurSamples,f.needsUpdate=!0,p.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new _n(i.x,i.y)),f.uniforms.shadow_pass.value=b.map.texture,f.uniforms.resolution.value=b.mapSize,f.uniforms.radius.value=b.radius,a.setRenderTarget(b.mapPass),a.clear(),a.renderBufferDirect(y,null,C,f,_,null),p.uniforms.shadow_pass.value=b.mapPass.texture,p.uniforms.resolution.value=b.mapSize,p.uniforms.radius.value=b.radius,a.setRenderTarget(b.map),a.clear(),a.renderBufferDirect(y,null,C,p,_,null)}function E(b,y,C,D){let R=null;const M=C.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(M!==void 0)R=M;else if(R=C.isPointLight===!0?c:s,a.localClippingEnabled&&y.clipShadows===!0&&Array.isArray(y.clippingPlanes)&&y.clippingPlanes.length!==0||y.displacementMap&&y.displacementScale!==0||y.alphaMap&&y.alphaTest>0||y.map&&y.alphaTest>0){const T=R.uuid,X=y.uuid;let q=l[T];q===void 0&&(q={},l[T]=q);let I=q[X];I===void 0&&(I=R.clone(),q[X]=I),R=I}if(R.visible=y.visible,R.wireframe=y.wireframe,D===ei?R.side=y.shadowSide!==null?y.shadowSide:y.side:R.side=y.shadowSide!==null?y.shadowSide:d[y.side],R.alphaMap=y.alphaMap,R.alphaTest=y.alphaTest,R.map=y.map,R.clipShadows=y.clipShadows,R.clippingPlanes=y.clippingPlanes,R.clipIntersection=y.clipIntersection,R.displacementMap=y.displacementMap,R.displacementScale=y.displacementScale,R.displacementBias=y.displacementBias,R.wireframeLinewidth=y.wireframeLinewidth,R.linewidth=y.linewidth,C.isPointLight===!0&&R.isMeshDistanceMaterial===!0){const T=a.properties.get(R);T.light=C}return R}function x(b,y,C,D,R){if(b.visible===!1)return;if(b.layers.test(y.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&R===ei)&&(!b.frustumCulled||n.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(C.matrixWorldInverse,b.matrixWorld);const X=t.update(b),q=b.material;if(Array.isArray(q)){const I=X.groups;for(let N=0,B=I.length;N<B;N++){const Q=I[N],et=q[Q.materialIndex];if(et&&et.visible){const j=E(b,et,D,R);a.renderBufferDirect(C,null,X,j,b,Q)}}}else if(q.visible){const I=E(b,q,D,R);a.renderBufferDirect(C,null,X,I,b,null)}}const T=b.children;for(let X=0,q=T.length;X<q;X++)x(T[X],y,C,D,R)}}function Hd(a,t,e){const n=e.isWebGL2;function i(){let L=!1;const H=new ie;let $=null;const ut=new ie(0,0,0,0);return{setMask:function(_t){$!==_t&&!L&&(a.colorMask(_t,_t,_t,_t),$=_t)},setLocked:function(_t){L=_t},setClear:function(_t,Vt,Ht,oe,Je){Je===!0&&(_t*=oe,Vt*=oe,Ht*=oe),H.set(_t,Vt,Ht,oe),ut.equals(H)===!1&&(a.clearColor(_t,Vt,Ht,oe),ut.copy(H))},reset:function(){L=!1,$=null,ut.set(-1,0,0,0)}}}function r(){let L=!1,H=null,$=null,ut=null;return{setTest:function(_t){_t?z(2929):bt(2929)},setMask:function(_t){H!==_t&&!L&&(a.depthMask(_t),H=_t)},setFunc:function(_t){if($!==_t){switch(_t){case _o:a.depthFunc(512);break;case vo:a.depthFunc(519);break;case xo:a.depthFunc(513);break;case Tr:a.depthFunc(515);break;case Mo:a.depthFunc(514);break;case So:a.depthFunc(518);break;case yo:a.depthFunc(516);break;case bo:a.depthFunc(517);break;default:a.depthFunc(515)}$=_t}},setLocked:function(_t){L=_t},setClear:function(_t){ut!==_t&&(a.clearDepth(_t),ut=_t)},reset:function(){L=!1,H=null,$=null,ut=null}}}function o(){let L=!1,H=null,$=null,ut=null,_t=null,Vt=null,Ht=null,oe=null,Je=null;return{setTest:function(Yt){L||(Yt?z(2960):bt(2960))},setMask:function(Yt){H!==Yt&&!L&&(a.stencilMask(Yt),H=Yt)},setFunc:function(Yt,Se,Ue){($!==Yt||ut!==Se||_t!==Ue)&&(a.stencilFunc(Yt,Se,Ue),$=Yt,ut=Se,_t=Ue)},setOp:function(Yt,Se,Ue){(Vt!==Yt||Ht!==Se||oe!==Ue)&&(a.stencilOp(Yt,Se,Ue),Vt=Yt,Ht=Se,oe=Ue)},setLocked:function(Yt){L=Yt},setClear:function(Yt){Je!==Yt&&(a.clearStencil(Yt),Je=Yt)},reset:function(){L=!1,H=null,$=null,ut=null,_t=null,Vt=null,Ht=null,oe=null,Je=null}}}const s=new i,c=new r,l=new o,h=new WeakMap,d=new WeakMap;let f={},p={},g=new WeakMap,_=[],m=null,u=!1,E=null,x=null,b=null,y=null,C=null,D=null,R=null,M=!1,T=null,X=null,q=null,I=null,N=null;const B=a.getParameter(35661);let Q=!1,et=0;const j=a.getParameter(7938);j.indexOf("WebGL")!==-1?(et=parseFloat(/^WebGL (\d)/.exec(j)[1]),Q=et>=1):j.indexOf("OpenGL ES")!==-1&&(et=parseFloat(/^OpenGL ES (\d)/.exec(j)[1]),Q=et>=2);let Z=null,nt={};const vt=a.getParameter(3088),ct=a.getParameter(2978),V=new ie().fromArray(vt),J=new ie().fromArray(ct);function st(L,H,$){const ut=new Uint8Array(4),_t=a.createTexture();a.bindTexture(L,_t),a.texParameteri(L,10241,9728),a.texParameteri(L,10240,9728);for(let Vt=0;Vt<$;Vt++)a.texImage2D(H+Vt,0,6408,1,1,0,6408,5121,ut);return _t}const lt={};lt[3553]=st(3553,3553,1),lt[34067]=st(34067,34069,6),s.setClear(0,0,0,1),c.setClear(1),l.setClear(0),z(2929),c.setFunc(Tr),kt(!1),jt(Zr),z(2884),Tt(tn);function z(L){f[L]!==!0&&(a.enable(L),f[L]=!0)}function bt(L){f[L]!==!1&&(a.disable(L),f[L]=!1)}function St(L,H){return p[L]!==H?(a.bindFramebuffer(L,H),p[L]=H,n&&(L===36009&&(p[36160]=H),L===36160&&(p[36009]=H)),!0):!1}function it(L,H){let $=_,ut=!1;if(L)if($=g.get(H),$===void 0&&($=[],g.set(H,$)),L.isWebGLMultipleRenderTargets){const _t=L.texture;if($.length!==_t.length||$[0]!==36064){for(let Vt=0,Ht=_t.length;Vt<Ht;Vt++)$[Vt]=36064+Vt;$.length=_t.length,ut=!0}}else $[0]!==36064&&($[0]=36064,ut=!0);else $[0]!==1029&&($[0]=1029,ut=!0);ut&&(e.isWebGL2?a.drawBuffers($):t.get("WEBGL_draw_buffers").drawBuffersWEBGL($))}function Mt(L){return m!==L?(a.useProgram(L),m=L,!0):!1}const Ft={[Bn]:32774,[so]:32778,[ao]:32779};if(n)Ft[ts]=32775,Ft[es]=32776;else{const L=t.get("EXT_blend_minmax");L!==null&&(Ft[ts]=L.MIN_EXT,Ft[es]=L.MAX_EXT)}const gt={[oo]:0,[lo]:1,[co]:768,[Sa]:770,[go]:776,[po]:774,[uo]:772,[ho]:769,[ya]:771,[mo]:775,[fo]:773};function Tt(L,H,$,ut,_t,Vt,Ht,oe){if(L===tn){u===!0&&(bt(3042),u=!1);return}if(u===!1&&(z(3042),u=!0),L!==ro){if(L!==E||oe!==M){if((x!==Bn||C!==Bn)&&(a.blendEquation(32774),x=Bn,C=Bn),oe)switch(L){case Vn:a.blendFuncSeparate(1,771,1,771);break;case Kr:a.blendFunc(1,1);break;case $r:a.blendFuncSeparate(0,769,0,1);break;case Qr:a.blendFuncSeparate(0,768,0,770);break;default:console.error("THREE.WebGLState: Invalid blending: ",L);break}else switch(L){case Vn:a.blendFuncSeparate(770,771,1,771);break;case Kr:a.blendFunc(770,1);break;case $r:a.blendFuncSeparate(0,769,0,1);break;case Qr:a.blendFunc(0,768);break;default:console.error("THREE.WebGLState: Invalid blending: ",L);break}b=null,y=null,D=null,R=null,E=L,M=oe}return}_t=_t||H,Vt=Vt||$,Ht=Ht||ut,(H!==x||_t!==C)&&(a.blendEquationSeparate(Ft[H],Ft[_t]),x=H,C=_t),($!==b||ut!==y||Vt!==D||Ht!==R)&&(a.blendFuncSeparate(gt[$],gt[ut],gt[Vt],gt[Ht]),b=$,y=ut,D=Vt,R=Ht),E=L,M=!1}function Wt(L,H){L.side===Xe?bt(2884):z(2884);let $=L.side===pe;H&&($=!$),kt($),L.blending===Vn&&L.transparent===!1?Tt(tn):Tt(L.blending,L.blendEquation,L.blendSrc,L.blendDst,L.blendEquationAlpha,L.blendSrcAlpha,L.blendDstAlpha,L.premultipliedAlpha),c.setFunc(L.depthFunc),c.setTest(L.depthTest),c.setMask(L.depthWrite),s.setMask(L.colorWrite);const ut=L.stencilWrite;l.setTest(ut),ut&&(l.setMask(L.stencilWriteMask),l.setFunc(L.stencilFunc,L.stencilRef,L.stencilFuncMask),l.setOp(L.stencilFail,L.stencilZFail,L.stencilZPass)),Nt(L.polygonOffset,L.polygonOffsetFactor,L.polygonOffsetUnits),L.alphaToCoverage===!0?z(32926):bt(32926)}function kt(L){T!==L&&(L?a.frontFace(2304):a.frontFace(2305),T=L)}function jt(L){L!==no?(z(2884),L!==X&&(L===Zr?a.cullFace(1029):L===io?a.cullFace(1028):a.cullFace(1032))):bt(2884),X=L}function qt(L){L!==q&&(Q&&a.lineWidth(L),q=L)}function Nt(L,H,$){L?(z(32823),(I!==H||N!==$)&&(a.polygonOffset(H,$),I=H,N=$)):bt(32823)}function Bt(L){L?z(3089):bt(3089)}function ae(L){L===void 0&&(L=33984+B-1),Z!==L&&(a.activeTexture(L),Z=L)}function w(L,H,$){$===void 0&&(Z===null?$=33984+B-1:$=Z);let ut=nt[$];ut===void 0&&(ut={type:void 0,texture:void 0},nt[$]=ut),(ut.type!==L||ut.texture!==H)&&(Z!==$&&(a.activeTexture($),Z=$),a.bindTexture(L,H||lt[L]),ut.type=L,ut.texture=H)}function v(){const L=nt[Z];L!==void 0&&L.type!==void 0&&(a.bindTexture(L.type,null),L.type=void 0,L.texture=void 0)}function G(){try{a.compressedTexImage2D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function K(){try{a.compressedTexImage3D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function tt(){try{a.texSubImage2D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function at(){try{a.texSubImage3D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function A(){try{a.compressedTexSubImage2D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function Y(){try{a.compressedTexSubImage3D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function F(){try{a.texStorage2D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function ot(){try{a.texStorage3D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function ft(){try{a.texImage2D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function mt(){try{a.texImage3D.apply(a,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function ht(L){V.equals(L)===!1&&(a.scissor(L.x,L.y,L.z,L.w),V.copy(L))}function pt(L){J.equals(L)===!1&&(a.viewport(L.x,L.y,L.z,L.w),J.copy(L))}function wt(L,H){let $=d.get(H);$===void 0&&($=new WeakMap,d.set(H,$));let ut=$.get(L);ut===void 0&&(ut=a.getUniformBlockIndex(H,L.name),$.set(L,ut))}function Lt(L,H){const ut=d.get(H).get(L);h.get(H)!==ut&&(a.uniformBlockBinding(H,ut,L.__bindingPointIndex),h.set(H,ut))}function Xt(){a.disable(3042),a.disable(2884),a.disable(2929),a.disable(32823),a.disable(3089),a.disable(2960),a.disable(32926),a.blendEquation(32774),a.blendFunc(1,0),a.blendFuncSeparate(1,0,1,0),a.colorMask(!0,!0,!0,!0),a.clearColor(0,0,0,0),a.depthMask(!0),a.depthFunc(513),a.clearDepth(1),a.stencilMask(4294967295),a.stencilFunc(519,0,4294967295),a.stencilOp(7680,7680,7680),a.clearStencil(0),a.cullFace(1029),a.frontFace(2305),a.polygonOffset(0,0),a.activeTexture(33984),a.bindFramebuffer(36160,null),n===!0&&(a.bindFramebuffer(36009,null),a.bindFramebuffer(36008,null)),a.useProgram(null),a.lineWidth(1),a.scissor(0,0,a.canvas.width,a.canvas.height),a.viewport(0,0,a.canvas.width,a.canvas.height),f={},Z=null,nt={},p={},g=new WeakMap,_=[],m=null,u=!1,E=null,x=null,b=null,y=null,C=null,D=null,R=null,M=!1,T=null,X=null,q=null,I=null,N=null,V.set(0,0,a.canvas.width,a.canvas.height),J.set(0,0,a.canvas.width,a.canvas.height),s.reset(),c.reset(),l.reset()}return{buffers:{color:s,depth:c,stencil:l},enable:z,disable:bt,bindFramebuffer:St,drawBuffers:it,useProgram:Mt,setBlending:Tt,setMaterial:Wt,setFlipSided:kt,setCullFace:jt,setLineWidth:qt,setPolygonOffset:Nt,setScissorTest:Bt,activeTexture:ae,bindTexture:w,unbindTexture:v,compressedTexImage2D:G,compressedTexImage3D:K,texImage2D:ft,texImage3D:mt,updateUBOMapping:wt,uniformBlockBinding:Lt,texStorage2D:F,texStorage3D:ot,texSubImage2D:tt,texSubImage3D:at,compressedTexSubImage2D:A,compressedTexSubImage3D:Y,scissor:ht,viewport:pt,reset:Xt}}function Wd(a,t,e,n,i,r,o){const s=i.isWebGL2,c=i.maxTextures,l=i.maxCubemapSize,h=i.maxTextureSize,d=i.maxSamples,f=t.has("WEBGL_multisampled_render_to_texture")?t.get("WEBGL_multisampled_render_to_texture"):null,p=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),g=new WeakMap;let _;const m=new WeakMap;let u=!1;try{u=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function E(w,v){return u?new OffscreenCanvas(w,v):Ni("canvas")}function x(w,v,G,K){let tt=1;if((w.width>K||w.height>K)&&(tt=K/Math.max(w.width,w.height)),tt<1||v===!0)if(typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&w instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&w instanceof ImageBitmap){const at=v?Ko:Math.floor,A=at(tt*w.width),Y=at(tt*w.height);_===void 0&&(_=E(A,Y));const F=G?E(A,Y):_;return F.width=A,F.height=Y,F.getContext("2d").drawImage(w,0,0,A,Y),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+w.width+"x"+w.height+") to ("+A+"x"+Y+")."),F}else return"data"in w&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+w.width+"x"+w.height+")."),w;return w}function b(w){return Cs(w.width)&&Cs(w.height)}function y(w){return s?!1:w.wrapS!==Re||w.wrapT!==Re||w.minFilter!==fe&&w.minFilter!==be}function C(w,v){return w.generateMipmaps&&v&&w.minFilter!==fe&&w.minFilter!==be}function D(w){a.generateMipmap(w)}function R(w,v,G,K,tt=!1){if(s===!1)return v;if(w!==null){if(a[w]!==void 0)return a[w];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+w+"'")}let at=v;return v===6403&&(G===5126&&(at=33326),G===5131&&(at=33325),G===5121&&(at=33321)),v===33319&&(G===5126&&(at=33328),G===5131&&(at=33327),G===5121&&(at=33323)),v===6408&&(G===5126&&(at=34836),G===5131&&(at=34842),G===5121&&(at=K===Gt&&tt===!1?35907:32856),G===32819&&(at=32854),G===32820&&(at=32855)),(at===33325||at===33326||at===33327||at===33328||at===34842||at===34836)&&t.get("EXT_color_buffer_float"),at}function M(w,v,G){return C(w,G)===!0||w.isFramebufferTexture&&w.minFilter!==fe&&w.minFilter!==be?Math.log2(Math.max(v.width,v.height))+1:w.mipmaps!==void 0&&w.mipmaps.length>0?w.mipmaps.length:w.isCompressedTexture&&Array.isArray(w.image)?v.mipmaps.length:1}function T(w){return w===fe||w===ns||w===Wi?9728:9729}function X(w){const v=w.target;v.removeEventListener("dispose",X),I(v),v.isVideoTexture&&g.delete(v)}function q(w){const v=w.target;v.removeEventListener("dispose",q),B(v)}function I(w){const v=n.get(w);if(v.__webglInit===void 0)return;const G=w.source,K=m.get(G);if(K){const tt=K[v.__cacheKey];tt.usedTimes--,tt.usedTimes===0&&N(w),Object.keys(K).length===0&&m.delete(G)}n.remove(w)}function N(w){const v=n.get(w);a.deleteTexture(v.__webglTexture);const G=w.source,K=m.get(G);delete K[v.__cacheKey],o.memory.textures--}function B(w){const v=w.texture,G=n.get(w),K=n.get(v);if(K.__webglTexture!==void 0&&(a.deleteTexture(K.__webglTexture),o.memory.textures--),w.depthTexture&&w.depthTexture.dispose(),w.isWebGLCubeRenderTarget)for(let tt=0;tt<6;tt++)a.deleteFramebuffer(G.__webglFramebuffer[tt]),G.__webglDepthbuffer&&a.deleteRenderbuffer(G.__webglDepthbuffer[tt]);else{if(a.deleteFramebuffer(G.__webglFramebuffer),G.__webglDepthbuffer&&a.deleteRenderbuffer(G.__webglDepthbuffer),G.__webglMultisampledFramebuffer&&a.deleteFramebuffer(G.__webglMultisampledFramebuffer),G.__webglColorRenderbuffer)for(let tt=0;tt<G.__webglColorRenderbuffer.length;tt++)G.__webglColorRenderbuffer[tt]&&a.deleteRenderbuffer(G.__webglColorRenderbuffer[tt]);G.__webglDepthRenderbuffer&&a.deleteRenderbuffer(G.__webglDepthRenderbuffer)}if(w.isWebGLMultipleRenderTargets)for(let tt=0,at=v.length;tt<at;tt++){const A=n.get(v[tt]);A.__webglTexture&&(a.deleteTexture(A.__webglTexture),o.memory.textures--),n.remove(v[tt])}n.remove(v),n.remove(w)}let Q=0;function et(){Q=0}function j(){const w=Q;return w>=c&&console.warn("THREE.WebGLTextures: Trying to use "+w+" texture units while this GPU supports only "+c),Q+=1,w}function Z(w){const v=[];return v.push(w.wrapS),v.push(w.wrapT),v.push(w.wrapR||0),v.push(w.magFilter),v.push(w.minFilter),v.push(w.anisotropy),v.push(w.internalFormat),v.push(w.format),v.push(w.type),v.push(w.generateMipmaps),v.push(w.premultiplyAlpha),v.push(w.flipY),v.push(w.unpackAlignment),v.push(w.encoding),v.join()}function nt(w,v){const G=n.get(w);if(w.isVideoTexture&&Bt(w),w.isRenderTargetTexture===!1&&w.version>0&&G.__version!==w.version){const K=w.image;if(K===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if(K.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{bt(G,w,v);return}}e.bindTexture(3553,G.__webglTexture,33984+v)}function vt(w,v){const G=n.get(w);if(w.version>0&&G.__version!==w.version){bt(G,w,v);return}e.bindTexture(35866,G.__webglTexture,33984+v)}function ct(w,v){const G=n.get(w);if(w.version>0&&G.__version!==w.version){bt(G,w,v);return}e.bindTexture(32879,G.__webglTexture,33984+v)}function V(w,v){const G=n.get(w);if(w.version>0&&G.__version!==w.version){St(G,w,v);return}e.bindTexture(34067,G.__webglTexture,33984+v)}const J={[Lr]:10497,[Re]:33071,[Pr]:33648},st={[fe]:9728,[ns]:9984,[Wi]:9986,[be]:9729,[Po]:9985,[si]:9987};function lt(w,v,G){if(G?(a.texParameteri(w,10242,J[v.wrapS]),a.texParameteri(w,10243,J[v.wrapT]),(w===32879||w===35866)&&a.texParameteri(w,32882,J[v.wrapR]),a.texParameteri(w,10240,st[v.magFilter]),a.texParameteri(w,10241,st[v.minFilter])):(a.texParameteri(w,10242,33071),a.texParameteri(w,10243,33071),(w===32879||w===35866)&&a.texParameteri(w,32882,33071),(v.wrapS!==Re||v.wrapT!==Re)&&console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.wrapS and Texture.wrapT should be set to THREE.ClampToEdgeWrapping."),a.texParameteri(w,10240,T(v.magFilter)),a.texParameteri(w,10241,T(v.minFilter)),v.minFilter!==fe&&v.minFilter!==be&&console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.minFilter should be set to THREE.NearestFilter or THREE.LinearFilter.")),t.has("EXT_texture_filter_anisotropic")===!0){const K=t.get("EXT_texture_filter_anisotropic");if(v.magFilter===fe||v.minFilter!==Wi&&v.minFilter!==si||v.type===fn&&t.has("OES_texture_float_linear")===!1||s===!1&&v.type===ai&&t.has("OES_texture_half_float_linear")===!1)return;(v.anisotropy>1||n.get(v).__currentAnisotropy)&&(a.texParameterf(w,K.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,i.getMaxAnisotropy())),n.get(v).__currentAnisotropy=v.anisotropy)}}function z(w,v){let G=!1;w.__webglInit===void 0&&(w.__webglInit=!0,v.addEventListener("dispose",X));const K=v.source;let tt=m.get(K);tt===void 0&&(tt={},m.set(K,tt));const at=Z(v);if(at!==w.__cacheKey){tt[at]===void 0&&(tt[at]={texture:a.createTexture(),usedTimes:0},o.memory.textures++,G=!0),tt[at].usedTimes++;const A=tt[w.__cacheKey];A!==void 0&&(tt[w.__cacheKey].usedTimes--,A.usedTimes===0&&N(v)),w.__cacheKey=at,w.__webglTexture=tt[at].texture}return G}function bt(w,v,G){let K=3553;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(K=35866),v.isData3DTexture&&(K=32879);const tt=z(w,v),at=v.source;e.bindTexture(K,w.__webglTexture,33984+G);const A=n.get(at);if(at.version!==A.__version||tt===!0){e.activeTexture(33984+G),a.pixelStorei(37440,v.flipY),a.pixelStorei(37441,v.premultiplyAlpha),a.pixelStorei(3317,v.unpackAlignment),a.pixelStorei(37443,0);const Y=y(v)&&b(v.image)===!1;let F=x(v.image,Y,!1,h);F=ae(v,F);const ot=b(F)||s,ft=r.convert(v.format,v.encoding);let mt=r.convert(v.type),ht=R(v.internalFormat,ft,mt,v.encoding,v.isVideoTexture);lt(K,v,ot);let pt;const wt=v.mipmaps,Lt=s&&v.isVideoTexture!==!0,Xt=A.__version===void 0||tt===!0,L=M(v,F,ot);if(v.isDepthTexture)ht=6402,s?v.type===fn?ht=36012:v.type===dn?ht=33190:v.type===kn?ht=35056:ht=33189:v.type===fn&&console.error("WebGLRenderer: Floating point depth texture requires WebGL2."),v.format===pn&&ht===6402&&v.type!==Ta&&v.type!==dn&&(console.warn("THREE.WebGLRenderer: Use UnsignedShortType or UnsignedIntType for DepthFormat DepthTexture."),v.type=dn,mt=r.convert(v.type)),v.format===Xn&&ht===6402&&(ht=34041,v.type!==kn&&(console.warn("THREE.WebGLRenderer: Use UnsignedInt248Type for DepthStencilFormat DepthTexture."),v.type=kn,mt=r.convert(v.type))),Xt&&(Lt?e.texStorage2D(3553,1,ht,F.width,F.height):e.texImage2D(3553,0,ht,F.width,F.height,0,ft,mt,null));else if(v.isDataTexture)if(wt.length>0&&ot){Lt&&Xt&&e.texStorage2D(3553,L,ht,wt[0].width,wt[0].height);for(let H=0,$=wt.length;H<$;H++)pt=wt[H],Lt?e.texSubImage2D(3553,H,0,0,pt.width,pt.height,ft,mt,pt.data):e.texImage2D(3553,H,ht,pt.width,pt.height,0,ft,mt,pt.data);v.generateMipmaps=!1}else Lt?(Xt&&e.texStorage2D(3553,L,ht,F.width,F.height),e.texSubImage2D(3553,0,0,0,F.width,F.height,ft,mt,F.data)):e.texImage2D(3553,0,ht,F.width,F.height,0,ft,mt,F.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){Lt&&Xt&&e.texStorage3D(35866,L,ht,wt[0].width,wt[0].height,F.depth);for(let H=0,$=wt.length;H<$;H++)pt=wt[H],v.format!==De?ft!==null?Lt?e.compressedTexSubImage3D(35866,H,0,0,0,pt.width,pt.height,F.depth,ft,pt.data,0,0):e.compressedTexImage3D(35866,H,ht,pt.width,pt.height,F.depth,0,pt.data,0,0):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Lt?e.texSubImage3D(35866,H,0,0,0,pt.width,pt.height,F.depth,ft,mt,pt.data):e.texImage3D(35866,H,ht,pt.width,pt.height,F.depth,0,ft,mt,pt.data)}else{Lt&&Xt&&e.texStorage2D(3553,L,ht,wt[0].width,wt[0].height);for(let H=0,$=wt.length;H<$;H++)pt=wt[H],v.format!==De?ft!==null?Lt?e.compressedTexSubImage2D(3553,H,0,0,pt.width,pt.height,ft,pt.data):e.compressedTexImage2D(3553,H,ht,pt.width,pt.height,0,pt.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Lt?e.texSubImage2D(3553,H,0,0,pt.width,pt.height,ft,mt,pt.data):e.texImage2D(3553,H,ht,pt.width,pt.height,0,ft,mt,pt.data)}else if(v.isDataArrayTexture)Lt?(Xt&&e.texStorage3D(35866,L,ht,F.width,F.height,F.depth),e.texSubImage3D(35866,0,0,0,0,F.width,F.height,F.depth,ft,mt,F.data)):e.texImage3D(35866,0,ht,F.width,F.height,F.depth,0,ft,mt,F.data);else if(v.isData3DTexture)Lt?(Xt&&e.texStorage3D(32879,L,ht,F.width,F.height,F.depth),e.texSubImage3D(32879,0,0,0,0,F.width,F.height,F.depth,ft,mt,F.data)):e.texImage3D(32879,0,ht,F.width,F.height,F.depth,0,ft,mt,F.data);else if(v.isFramebufferTexture){if(Xt)if(Lt)e.texStorage2D(3553,L,ht,F.width,F.height);else{let H=F.width,$=F.height;for(let ut=0;ut<L;ut++)e.texImage2D(3553,ut,ht,H,$,0,ft,mt,null),H>>=1,$>>=1}}else if(wt.length>0&&ot){Lt&&Xt&&e.texStorage2D(3553,L,ht,wt[0].width,wt[0].height);for(let H=0,$=wt.length;H<$;H++)pt=wt[H],Lt?e.texSubImage2D(3553,H,0,0,ft,mt,pt):e.texImage2D(3553,H,ht,ft,mt,pt);v.generateMipmaps=!1}else Lt?(Xt&&e.texStorage2D(3553,L,ht,F.width,F.height),e.texSubImage2D(3553,0,0,0,ft,mt,F)):e.texImage2D(3553,0,ht,ft,mt,F);C(v,ot)&&D(K),A.__version=at.version,v.onUpdate&&v.onUpdate(v)}w.__version=v.version}function St(w,v,G){if(v.image.length!==6)return;const K=z(w,v),tt=v.source;e.bindTexture(34067,w.__webglTexture,33984+G);const at=n.get(tt);if(tt.version!==at.__version||K===!0){e.activeTexture(33984+G),a.pixelStorei(37440,v.flipY),a.pixelStorei(37441,v.premultiplyAlpha),a.pixelStorei(3317,v.unpackAlignment),a.pixelStorei(37443,0);const A=v.isCompressedTexture||v.image[0].isCompressedTexture,Y=v.image[0]&&v.image[0].isDataTexture,F=[];for(let H=0;H<6;H++)!A&&!Y?F[H]=x(v.image[H],!1,!0,l):F[H]=Y?v.image[H].image:v.image[H],F[H]=ae(v,F[H]);const ot=F[0],ft=b(ot)||s,mt=r.convert(v.format,v.encoding),ht=r.convert(v.type),pt=R(v.internalFormat,mt,ht,v.encoding),wt=s&&v.isVideoTexture!==!0,Lt=at.__version===void 0||K===!0;let Xt=M(v,ot,ft);lt(34067,v,ft);let L;if(A){wt&&Lt&&e.texStorage2D(34067,Xt,pt,ot.width,ot.height);for(let H=0;H<6;H++){L=F[H].mipmaps;for(let $=0;$<L.length;$++){const ut=L[$];v.format!==De?mt!==null?wt?e.compressedTexSubImage2D(34069+H,$,0,0,ut.width,ut.height,mt,ut.data):e.compressedTexImage2D(34069+H,$,pt,ut.width,ut.height,0,ut.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):wt?e.texSubImage2D(34069+H,$,0,0,ut.width,ut.height,mt,ht,ut.data):e.texImage2D(34069+H,$,pt,ut.width,ut.height,0,mt,ht,ut.data)}}}else{L=v.mipmaps,wt&&Lt&&(L.length>0&&Xt++,e.texStorage2D(34067,Xt,pt,F[0].width,F[0].height));for(let H=0;H<6;H++)if(Y){wt?e.texSubImage2D(34069+H,0,0,0,F[H].width,F[H].height,mt,ht,F[H].data):e.texImage2D(34069+H,0,pt,F[H].width,F[H].height,0,mt,ht,F[H].data);for(let $=0;$<L.length;$++){const _t=L[$].image[H].image;wt?e.texSubImage2D(34069+H,$+1,0,0,_t.width,_t.height,mt,ht,_t.data):e.texImage2D(34069+H,$+1,pt,_t.width,_t.height,0,mt,ht,_t.data)}}else{wt?e.texSubImage2D(34069+H,0,0,0,mt,ht,F[H]):e.texImage2D(34069+H,0,pt,mt,ht,F[H]);for(let $=0;$<L.length;$++){const ut=L[$];wt?e.texSubImage2D(34069+H,$+1,0,0,mt,ht,ut.image[H]):e.texImage2D(34069+H,$+1,pt,mt,ht,ut.image[H])}}}C(v,ft)&&D(34067),at.__version=tt.version,v.onUpdate&&v.onUpdate(v)}w.__version=v.version}function it(w,v,G,K,tt){const at=r.convert(G.format,G.encoding),A=r.convert(G.type),Y=R(G.internalFormat,at,A,G.encoding);n.get(v).__hasExternalTextures||(tt===32879||tt===35866?e.texImage3D(tt,0,Y,v.width,v.height,v.depth,0,at,A,null):e.texImage2D(tt,0,Y,v.width,v.height,0,at,A,null)),e.bindFramebuffer(36160,w),Nt(v)?f.framebufferTexture2DMultisampleEXT(36160,K,tt,n.get(G).__webglTexture,0,qt(v)):(tt===3553||tt>=34069&&tt<=34074)&&a.framebufferTexture2D(36160,K,tt,n.get(G).__webglTexture,0),e.bindFramebuffer(36160,null)}function Mt(w,v,G){if(a.bindRenderbuffer(36161,w),v.depthBuffer&&!v.stencilBuffer){let K=33189;if(G||Nt(v)){const tt=v.depthTexture;tt&&tt.isDepthTexture&&(tt.type===fn?K=36012:tt.type===dn&&(K=33190));const at=qt(v);Nt(v)?f.renderbufferStorageMultisampleEXT(36161,at,K,v.width,v.height):a.renderbufferStorageMultisample(36161,at,K,v.width,v.height)}else a.renderbufferStorage(36161,K,v.width,v.height);a.framebufferRenderbuffer(36160,36096,36161,w)}else if(v.depthBuffer&&v.stencilBuffer){const K=qt(v);G&&Nt(v)===!1?a.renderbufferStorageMultisample(36161,K,35056,v.width,v.height):Nt(v)?f.renderbufferStorageMultisampleEXT(36161,K,35056,v.width,v.height):a.renderbufferStorage(36161,34041,v.width,v.height),a.framebufferRenderbuffer(36160,33306,36161,w)}else{const K=v.isWebGLMultipleRenderTargets===!0?v.texture:[v.texture];for(let tt=0;tt<K.length;tt++){const at=K[tt],A=r.convert(at.format,at.encoding),Y=r.convert(at.type),F=R(at.internalFormat,A,Y,at.encoding),ot=qt(v);G&&Nt(v)===!1?a.renderbufferStorageMultisample(36161,ot,F,v.width,v.height):Nt(v)?f.renderbufferStorageMultisampleEXT(36161,ot,F,v.width,v.height):a.renderbufferStorage(36161,F,v.width,v.height)}}a.bindRenderbuffer(36161,null)}function Ft(w,v){if(v&&v.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(e.bindFramebuffer(36160,w),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");(!n.get(v.depthTexture).__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),nt(v.depthTexture,0);const K=n.get(v.depthTexture).__webglTexture,tt=qt(v);if(v.depthTexture.format===pn)Nt(v)?f.framebufferTexture2DMultisampleEXT(36160,36096,3553,K,0,tt):a.framebufferTexture2D(36160,36096,3553,K,0);else if(v.depthTexture.format===Xn)Nt(v)?f.framebufferTexture2DMultisampleEXT(36160,33306,3553,K,0,tt):a.framebufferTexture2D(36160,33306,3553,K,0);else throw new Error("Unknown depthTexture format")}function gt(w){const v=n.get(w),G=w.isWebGLCubeRenderTarget===!0;if(w.depthTexture&&!v.__autoAllocateDepthBuffer){if(G)throw new Error("target.depthTexture not supported in Cube render targets");Ft(v.__webglFramebuffer,w)}else if(G){v.__webglDepthbuffer=[];for(let K=0;K<6;K++)e.bindFramebuffer(36160,v.__webglFramebuffer[K]),v.__webglDepthbuffer[K]=a.createRenderbuffer(),Mt(v.__webglDepthbuffer[K],w,!1)}else e.bindFramebuffer(36160,v.__webglFramebuffer),v.__webglDepthbuffer=a.createRenderbuffer(),Mt(v.__webglDepthbuffer,w,!1);e.bindFramebuffer(36160,null)}function Tt(w,v,G){const K=n.get(w);v!==void 0&&it(K.__webglFramebuffer,w,w.texture,36064,3553),G!==void 0&&gt(w)}function Wt(w){const v=w.texture,G=n.get(w),K=n.get(v);w.addEventListener("dispose",q),w.isWebGLMultipleRenderTargets!==!0&&(K.__webglTexture===void 0&&(K.__webglTexture=a.createTexture()),K.__version=v.version,o.memory.textures++);const tt=w.isWebGLCubeRenderTarget===!0,at=w.isWebGLMultipleRenderTargets===!0,A=b(w)||s;if(tt){G.__webglFramebuffer=[];for(let Y=0;Y<6;Y++)G.__webglFramebuffer[Y]=a.createFramebuffer()}else{if(G.__webglFramebuffer=a.createFramebuffer(),at)if(i.drawBuffers){const Y=w.texture;for(let F=0,ot=Y.length;F<ot;F++){const ft=n.get(Y[F]);ft.__webglTexture===void 0&&(ft.__webglTexture=a.createTexture(),o.memory.textures++)}}else console.warn("THREE.WebGLRenderer: WebGLMultipleRenderTargets can only be used with WebGL2 or WEBGL_draw_buffers extension.");if(s&&w.samples>0&&Nt(w)===!1){const Y=at?v:[v];G.__webglMultisampledFramebuffer=a.createFramebuffer(),G.__webglColorRenderbuffer=[],e.bindFramebuffer(36160,G.__webglMultisampledFramebuffer);for(let F=0;F<Y.length;F++){const ot=Y[F];G.__webglColorRenderbuffer[F]=a.createRenderbuffer(),a.bindRenderbuffer(36161,G.__webglColorRenderbuffer[F]);const ft=r.convert(ot.format,ot.encoding),mt=r.convert(ot.type),ht=R(ot.internalFormat,ft,mt,ot.encoding,w.isXRRenderTarget===!0),pt=qt(w);a.renderbufferStorageMultisample(36161,pt,ht,w.width,w.height),a.framebufferRenderbuffer(36160,36064+F,36161,G.__webglColorRenderbuffer[F])}a.bindRenderbuffer(36161,null),w.depthBuffer&&(G.__webglDepthRenderbuffer=a.createRenderbuffer(),Mt(G.__webglDepthRenderbuffer,w,!0)),e.bindFramebuffer(36160,null)}}if(tt){e.bindTexture(34067,K.__webglTexture),lt(34067,v,A);for(let Y=0;Y<6;Y++)it(G.__webglFramebuffer[Y],w,v,36064,34069+Y);C(v,A)&&D(34067),e.unbindTexture()}else if(at){const Y=w.texture;for(let F=0,ot=Y.length;F<ot;F++){const ft=Y[F],mt=n.get(ft);e.bindTexture(3553,mt.__webglTexture),lt(3553,ft,A),it(G.__webglFramebuffer,w,ft,36064+F,3553),C(ft,A)&&D(3553)}e.unbindTexture()}else{let Y=3553;(w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(s?Y=w.isWebGL3DRenderTarget?32879:35866:console.error("THREE.WebGLTextures: THREE.Data3DTexture and THREE.DataArrayTexture only supported with WebGL2.")),e.bindTexture(Y,K.__webglTexture),lt(Y,v,A),it(G.__webglFramebuffer,w,v,36064,Y),C(v,A)&&D(Y),e.unbindTexture()}w.depthBuffer&&gt(w)}function kt(w){const v=b(w)||s,G=w.isWebGLMultipleRenderTargets===!0?w.texture:[w.texture];for(let K=0,tt=G.length;K<tt;K++){const at=G[K];if(C(at,v)){const A=w.isWebGLCubeRenderTarget?34067:3553,Y=n.get(at).__webglTexture;e.bindTexture(A,Y),D(A),e.unbindTexture()}}}function jt(w){if(s&&w.samples>0&&Nt(w)===!1){const v=w.isWebGLMultipleRenderTargets?w.texture:[w.texture],G=w.width,K=w.height;let tt=16384;const at=[],A=w.stencilBuffer?33306:36096,Y=n.get(w),F=w.isWebGLMultipleRenderTargets===!0;if(F)for(let ot=0;ot<v.length;ot++)e.bindFramebuffer(36160,Y.__webglMultisampledFramebuffer),a.framebufferRenderbuffer(36160,36064+ot,36161,null),e.bindFramebuffer(36160,Y.__webglFramebuffer),a.framebufferTexture2D(36009,36064+ot,3553,null,0);e.bindFramebuffer(36008,Y.__webglMultisampledFramebuffer),e.bindFramebuffer(36009,Y.__webglFramebuffer);for(let ot=0;ot<v.length;ot++){at.push(36064+ot),w.depthBuffer&&at.push(A);const ft=Y.__ignoreDepthValues!==void 0?Y.__ignoreDepthValues:!1;if(ft===!1&&(w.depthBuffer&&(tt|=256),w.stencilBuffer&&(tt|=1024)),F&&a.framebufferRenderbuffer(36008,36064,36161,Y.__webglColorRenderbuffer[ot]),ft===!0&&(a.invalidateFramebuffer(36008,[A]),a.invalidateFramebuffer(36009,[A])),F){const mt=n.get(v[ot]).__webglTexture;a.framebufferTexture2D(36009,36064,3553,mt,0)}a.blitFramebuffer(0,0,G,K,0,0,G,K,tt,9728),p&&a.invalidateFramebuffer(36008,at)}if(e.bindFramebuffer(36008,null),e.bindFramebuffer(36009,null),F)for(let ot=0;ot<v.length;ot++){e.bindFramebuffer(36160,Y.__webglMultisampledFramebuffer),a.framebufferRenderbuffer(36160,36064+ot,36161,Y.__webglColorRenderbuffer[ot]);const ft=n.get(v[ot]).__webglTexture;e.bindFramebuffer(36160,Y.__webglFramebuffer),a.framebufferTexture2D(36009,36064+ot,3553,ft,0)}e.bindFramebuffer(36009,Y.__webglMultisampledFramebuffer)}}function qt(w){return Math.min(d,w.samples)}function Nt(w){const v=n.get(w);return s&&w.samples>0&&t.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function Bt(w){const v=o.render.frame;g.get(w)!==v&&(g.set(w,v),w.update())}function ae(w,v){const G=w.encoding,K=w.format,tt=w.type;return w.isCompressedTexture===!0||w.isVideoTexture===!0||w.format===Rr||G!==gn&&(G===Gt?s===!1?t.has("EXT_sRGB")===!0&&K===De?(w.format=Rr,w.minFilter=be,w.generateMipmaps=!1):v=Pa.sRGBToLinear(v):(K!==De||tt!==mn)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture encoding:",G)),v}this.allocateTextureUnit=j,this.resetTextureUnits=et,this.setTexture2D=nt,this.setTexture2DArray=vt,this.setTexture3D=ct,this.setTextureCube=V,this.rebindTextures=Tt,this.setupRenderTarget=Wt,this.updateRenderTargetMipmap=kt,this.updateMultisampleRenderTarget=jt,this.setupDepthRenderbuffer=gt,this.setupFrameBufferTexture=it,this.useMultisampledRTT=Nt}function qd(a,t,e){const n=e.isWebGL2;function i(r,o=null){let s;if(r===mn)return 5121;if(r===Uo)return 32819;if(r===No)return 32820;if(r===Ro)return 5120;if(r===Do)return 5122;if(r===Ta)return 5123;if(r===Io)return 5124;if(r===dn)return 5125;if(r===fn)return 5126;if(r===ai)return n?5131:(s=t.get("OES_texture_half_float"),s!==null?s.HALF_FLOAT_OES:null);if(r===Oo)return 6406;if(r===De)return 6408;if(r===Fo)return 6409;if(r===zo)return 6410;if(r===pn)return 6402;if(r===Xn)return 34041;if(r===Rr)return s=t.get("EXT_sRGB"),s!==null?s.SRGB_ALPHA_EXT:null;if(r===Bo)return 6403;if(r===Go)return 36244;if(r===Vo)return 33319;if(r===ko)return 33320;if(r===Ho)return 36249;if(r===qi||r===Xi||r===Yi||r===ji)if(o===Gt)if(s=t.get("WEBGL_compressed_texture_s3tc_srgb"),s!==null){if(r===qi)return s.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(r===Xi)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(r===Yi)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(r===ji)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(s=t.get("WEBGL_compressed_texture_s3tc"),s!==null){if(r===qi)return s.COMPRESSED_RGB_S3TC_DXT1_EXT;if(r===Xi)return s.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(r===Yi)return s.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(r===ji)return s.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(r===is||r===rs||r===ss||r===as)if(s=t.get("WEBGL_compressed_texture_pvrtc"),s!==null){if(r===is)return s.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(r===rs)return s.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(r===ss)return s.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(r===as)return s.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(r===Wo)return s=t.get("WEBGL_compressed_texture_etc1"),s!==null?s.COMPRESSED_RGB_ETC1_WEBGL:null;if(r===os||r===ls)if(s=t.get("WEBGL_compressed_texture_etc"),s!==null){if(r===os)return o===Gt?s.COMPRESSED_SRGB8_ETC2:s.COMPRESSED_RGB8_ETC2;if(r===ls)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:s.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(r===cs||r===hs||r===us||r===ds||r===fs||r===ps||r===ms||r===gs||r===_s||r===vs||r===xs||r===Ms||r===Ss||r===ys)if(s=t.get("WEBGL_compressed_texture_astc"),s!==null){if(r===cs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:s.COMPRESSED_RGBA_ASTC_4x4_KHR;if(r===hs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:s.COMPRESSED_RGBA_ASTC_5x4_KHR;if(r===us)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:s.COMPRESSED_RGBA_ASTC_5x5_KHR;if(r===ds)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:s.COMPRESSED_RGBA_ASTC_6x5_KHR;if(r===fs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:s.COMPRESSED_RGBA_ASTC_6x6_KHR;if(r===ps)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:s.COMPRESSED_RGBA_ASTC_8x5_KHR;if(r===ms)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:s.COMPRESSED_RGBA_ASTC_8x6_KHR;if(r===gs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:s.COMPRESSED_RGBA_ASTC_8x8_KHR;if(r===_s)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:s.COMPRESSED_RGBA_ASTC_10x5_KHR;if(r===vs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:s.COMPRESSED_RGBA_ASTC_10x6_KHR;if(r===xs)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:s.COMPRESSED_RGBA_ASTC_10x8_KHR;if(r===Ms)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:s.COMPRESSED_RGBA_ASTC_10x10_KHR;if(r===Ss)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:s.COMPRESSED_RGBA_ASTC_12x10_KHR;if(r===ys)return o===Gt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:s.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(r===Ji)if(s=t.get("EXT_texture_compression_bptc"),s!==null){if(r===Ji)return o===Gt?s.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:s.COMPRESSED_RGBA_BPTC_UNORM_EXT}else return null;if(r===qo||r===bs||r===ws||r===Es)if(s=t.get("EXT_texture_compression_rgtc"),s!==null){if(r===Ji)return s.COMPRESSED_RED_RGTC1_EXT;if(r===bs)return s.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(r===ws)return s.COMPRESSED_RED_GREEN_RGTC2_EXT;if(r===Es)return s.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return r===kn?n?34042:(s=t.get("WEBGL_depth_texture"),s!==null?s.UNSIGNED_INT_24_8_WEBGL:null):a[r]!==void 0?a[r]:null}return{convert:i}}class Xd extends we{constructor(t=[]){super(),this.isArrayCamera=!0,this.cameras=t}}class Ye extends re{constructor(){super(),this.isGroup=!0,this.type="Group"}}const Yd={type:"move"};class xr{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Ye,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Ye,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new P,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new P),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Ye,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new P,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new P),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){const e=this._hand;if(e)for(const n of t.hand.values())this._getHandJoint(e,n)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,n){let i=null,r=null,o=null;const s=this._targetRay,c=this._grip,l=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(l&&t.hand){o=!0;for(const _ of t.hand.values()){const m=e.getJointPose(_,n),u=this._getHandJoint(l,_);m!==null&&(u.matrix.fromArray(m.transform.matrix),u.matrix.decompose(u.position,u.rotation,u.scale),u.jointRadius=m.radius),u.visible=m!==null}const h=l.joints["index-finger-tip"],d=l.joints["thumb-tip"],f=h.position.distanceTo(d.position),p=.02,g=.005;l.inputState.pinching&&f>p+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!l.inputState.pinching&&f<=p-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else c!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1));s!==null&&(i=e.getPose(t.targetRaySpace,n),i===null&&r!==null&&(i=r),i!==null&&(s.matrix.fromArray(i.transform.matrix),s.matrix.decompose(s.position,s.rotation,s.scale),i.linearVelocity?(s.hasLinearVelocity=!0,s.linearVelocity.copy(i.linearVelocity)):s.hasLinearVelocity=!1,i.angularVelocity?(s.hasAngularVelocity=!0,s.angularVelocity.copy(i.angularVelocity)):s.hasAngularVelocity=!1,this.dispatchEvent(Yd)))}return s!==null&&(s.visible=i!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=o!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){const n=new Ye;n.matrixAutoUpdate=!1,n.visible=!1,t.joints[e.jointName]=n,t.add(n)}return t.joints[e.jointName]}}class jd extends Me{constructor(t,e,n,i,r,o,s,c,l,h){if(h=h!==void 0?h:pn,h!==pn&&h!==Xn)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");n===void 0&&h===pn&&(n=dn),n===void 0&&h===Xn&&(n=kn),super(null,i,r,o,s,c,h,n,l),this.isDepthTexture=!0,this.image={width:t,height:e},this.magFilter=s!==void 0?s:fe,this.minFilter=c!==void 0?c:fe,this.flipY=!1,this.generateMipmaps=!1}}class Jd extends Mn{constructor(t,e){super();const n=this;let i=null,r=1,o=null,s="local-floor",c=1,l=null,h=null,d=null,f=null,p=null,g=null;const _=e.getContextAttributes();let m=null,u=null;const E=[],x=[],b=new Set,y=new Map,C=new we;C.layers.enable(1),C.viewport=new ie;const D=new we;D.layers.enable(2),D.viewport=new ie;const R=[C,D],M=new Xd;M.layers.enable(1),M.layers.enable(2);let T=null,X=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(V){let J=E[V];return J===void 0&&(J=new xr,E[V]=J),J.getTargetRaySpace()},this.getControllerGrip=function(V){let J=E[V];return J===void 0&&(J=new xr,E[V]=J),J.getGripSpace()},this.getHand=function(V){let J=E[V];return J===void 0&&(J=new xr,E[V]=J),J.getHandSpace()};function q(V){const J=x.indexOf(V.inputSource);if(J===-1)return;const st=E[J];st!==void 0&&st.dispatchEvent({type:V.type,data:V.inputSource})}function I(){i.removeEventListener("select",q),i.removeEventListener("selectstart",q),i.removeEventListener("selectend",q),i.removeEventListener("squeeze",q),i.removeEventListener("squeezestart",q),i.removeEventListener("squeezeend",q),i.removeEventListener("end",I),i.removeEventListener("inputsourceschange",N);for(let V=0;V<E.length;V++){const J=x[V];J!==null&&(x[V]=null,E[V].disconnect(J))}T=null,X=null,t.setRenderTarget(m),p=null,f=null,d=null,i=null,u=null,ct.stop(),n.isPresenting=!1,n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(V){r=V,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(V){s=V,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||o},this.setReferenceSpace=function(V){l=V},this.getBaseLayer=function(){return f!==null?f:p},this.getBinding=function(){return d},this.getFrame=function(){return g},this.getSession=function(){return i},this.setSession=async function(V){if(i=V,i!==null){if(m=t.getRenderTarget(),i.addEventListener("select",q),i.addEventListener("selectstart",q),i.addEventListener("selectend",q),i.addEventListener("squeeze",q),i.addEventListener("squeezestart",q),i.addEventListener("squeezeend",q),i.addEventListener("end",I),i.addEventListener("inputsourceschange",N),_.xrCompatible!==!0&&await e.makeXRCompatible(),i.renderState.layers===void 0||t.capabilities.isWebGL2===!1){const J={antialias:i.renderState.layers===void 0?_.antialias:!0,alpha:_.alpha,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};p=new XRWebGLLayer(i,e,J),i.updateRenderState({baseLayer:p}),u=new _n(p.framebufferWidth,p.framebufferHeight,{format:De,type:mn,encoding:t.outputEncoding,stencilBuffer:_.stencil})}else{let J=null,st=null,lt=null;_.depth&&(lt=_.stencil?35056:33190,J=_.stencil?Xn:pn,st=_.stencil?kn:dn);const z={colorFormat:32856,depthFormat:lt,scaleFactor:r};d=new XRWebGLBinding(i,e),f=d.createProjectionLayer(z),i.updateRenderState({layers:[f]}),u=new _n(f.textureWidth,f.textureHeight,{format:De,type:mn,depthTexture:new jd(f.textureWidth,f.textureHeight,st,void 0,void 0,void 0,void 0,void 0,void 0,J),stencilBuffer:_.stencil,encoding:t.outputEncoding,samples:_.antialias?4:0});const bt=t.properties.get(u);bt.__ignoreDepthValues=f.ignoreDepthValues}u.isXRRenderTarget=!0,this.setFoveation(c),l=null,o=await i.requestReferenceSpace(s),ct.setContext(i),ct.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}};function N(V){for(let J=0;J<V.removed.length;J++){const st=V.removed[J],lt=x.indexOf(st);lt>=0&&(x[lt]=null,E[lt].disconnect(st))}for(let J=0;J<V.added.length;J++){const st=V.added[J];let lt=x.indexOf(st);if(lt===-1){for(let bt=0;bt<E.length;bt++)if(bt>=x.length){x.push(st),lt=bt;break}else if(x[bt]===null){x[bt]=st,lt=bt;break}if(lt===-1)break}const z=E[lt];z&&z.connect(st)}}const B=new P,Q=new P;function et(V,J,st){B.setFromMatrixPosition(J.matrixWorld),Q.setFromMatrixPosition(st.matrixWorld);const lt=B.distanceTo(Q),z=J.projectionMatrix.elements,bt=st.projectionMatrix.elements,St=z[14]/(z[10]-1),it=z[14]/(z[10]+1),Mt=(z[9]+1)/z[5],Ft=(z[9]-1)/z[5],gt=(z[8]-1)/z[0],Tt=(bt[8]+1)/bt[0],Wt=St*gt,kt=St*Tt,jt=lt/(-gt+Tt),qt=jt*-gt;J.matrixWorld.decompose(V.position,V.quaternion,V.scale),V.translateX(qt),V.translateZ(jt),V.matrixWorld.compose(V.position,V.quaternion,V.scale),V.matrixWorldInverse.copy(V.matrixWorld).invert();const Nt=St+jt,Bt=it+jt,ae=Wt-qt,w=kt+(lt-qt),v=Mt*it/Bt*Nt,G=Ft*it/Bt*Nt;V.projectionMatrix.makePerspective(ae,w,v,G,Nt,Bt),V.projectionMatrixInverse.copy(V.projectionMatrix).invert()}function j(V,J){J===null?V.matrixWorld.copy(V.matrix):V.matrixWorld.multiplyMatrices(J.matrixWorld,V.matrix),V.matrixWorldInverse.copy(V.matrixWorld).invert()}this.updateCamera=function(V){if(i===null)return;M.near=D.near=C.near=V.near,M.far=D.far=C.far=V.far,(T!==M.near||X!==M.far)&&(i.updateRenderState({depthNear:M.near,depthFar:M.far}),T=M.near,X=M.far);const J=V.parent,st=M.cameras;j(M,J);for(let lt=0;lt<st.length;lt++)j(st[lt],J);st.length===2?et(M,C,D):M.projectionMatrix.copy(C.projectionMatrix),Z(V,M,J)};function Z(V,J,st){st===null?V.matrix.copy(J.matrixWorld):(V.matrix.copy(st.matrixWorld),V.matrix.invert(),V.matrix.multiply(J.matrixWorld)),V.matrix.decompose(V.position,V.quaternion,V.scale),V.updateMatrixWorld(!0);const lt=V.children;for(let z=0,bt=lt.length;z<bt;z++)lt[z].updateMatrixWorld(!0);V.projectionMatrix.copy(J.projectionMatrix),V.projectionMatrixInverse.copy(J.projectionMatrixInverse),V.isPerspectiveCamera&&(V.fov=Dr*2*Math.atan(1/V.projectionMatrix.elements[5]),V.zoom=1)}this.getCamera=function(){return M},this.getFoveation=function(){if(!(f===null&&p===null))return c},this.setFoveation=function(V){c=V,f!==null&&(f.fixedFoveation=V),p!==null&&p.fixedFoveation!==void 0&&(p.fixedFoveation=V)},this.getPlanes=function(){return b};let nt=null;function vt(V,J){if(h=J.getViewerPose(l||o),g=J,h!==null){const st=h.views;p!==null&&(t.setRenderTargetFramebuffer(u,p.framebuffer),t.setRenderTarget(u));let lt=!1;st.length!==M.cameras.length&&(M.cameras.length=0,lt=!0);for(let z=0;z<st.length;z++){const bt=st[z];let St=null;if(p!==null)St=p.getViewport(bt);else{const Mt=d.getViewSubImage(f,bt);St=Mt.viewport,z===0&&(t.setRenderTargetTextures(u,Mt.colorTexture,f.ignoreDepthValues?void 0:Mt.depthStencilTexture),t.setRenderTarget(u))}let it=R[z];it===void 0&&(it=new we,it.layers.enable(z),it.viewport=new ie,R[z]=it),it.matrix.fromArray(bt.transform.matrix),it.matrix.decompose(it.position,it.quaternion,it.scale),it.projectionMatrix.fromArray(bt.projectionMatrix),it.projectionMatrixInverse.copy(it.projectionMatrix).invert(),it.viewport.set(St.x,St.y,St.width,St.height),z===0&&(M.matrix.copy(it.matrix),M.matrix.decompose(M.position,M.quaternion,M.scale)),lt===!0&&M.cameras.push(it)}}for(let st=0;st<E.length;st++){const lt=x[st],z=E[st];lt!==null&&z!==void 0&&z.update(lt,J,l||o)}if(nt&&nt(V,J),J.detectedPlanes){n.dispatchEvent({type:"planesdetected",data:J.detectedPlanes});let st=null;for(const lt of b)J.detectedPlanes.has(lt)||(st===null&&(st=[]),st.push(lt));if(st!==null)for(const lt of st)b.delete(lt),y.delete(lt),n.dispatchEvent({type:"planeremoved",data:lt});for(const lt of J.detectedPlanes)if(!b.has(lt))b.add(lt),y.set(lt,J.lastChangedTime),n.dispatchEvent({type:"planeadded",data:lt});else{const z=y.get(lt);lt.lastChangedTime>z&&(y.set(lt,lt.lastChangedTime),n.dispatchEvent({type:"planechanged",data:lt}))}}g=null}const ct=new Va;ct.setAnimationLoop(vt),this.setAnimationLoop=function(V){nt=V},this.dispose=function(){}}}function Zd(a,t){function e(m,u){m.matrixAutoUpdate===!0&&m.updateMatrix(),u.value.copy(m.matrix)}function n(m,u){u.color.getRGB(m.fogColor.value,za(a)),u.isFog?(m.fogNear.value=u.near,m.fogFar.value=u.far):u.isFogExp2&&(m.fogDensity.value=u.density)}function i(m,u,E,x,b){u.isMeshBasicMaterial||u.isMeshLambertMaterial?r(m,u):u.isMeshToonMaterial?(r(m,u),d(m,u)):u.isMeshPhongMaterial?(r(m,u),h(m,u)):u.isMeshStandardMaterial?(r(m,u),f(m,u),u.isMeshPhysicalMaterial&&p(m,u,b)):u.isMeshMatcapMaterial?(r(m,u),g(m,u)):u.isMeshDepthMaterial?r(m,u):u.isMeshDistanceMaterial?(r(m,u),_(m,u)):u.isMeshNormalMaterial?r(m,u):u.isLineBasicMaterial?(o(m,u),u.isLineDashedMaterial&&s(m,u)):u.isPointsMaterial?c(m,u,E,x):u.isSpriteMaterial?l(m,u):u.isShadowMaterial?(m.color.value.copy(u.color),m.opacity.value=u.opacity):u.isShaderMaterial&&(u.uniformsNeedUpdate=!1)}function r(m,u){m.opacity.value=u.opacity,u.color&&m.diffuse.value.copy(u.color),u.emissive&&m.emissive.value.copy(u.emissive).multiplyScalar(u.emissiveIntensity),u.map&&(m.map.value=u.map,e(u.map,m.mapTransform)),u.alphaMap&&(m.alphaMap.value=u.alphaMap,e(u.alphaMap,m.alphaMapTransform)),u.bumpMap&&(m.bumpMap.value=u.bumpMap,e(u.bumpMap,m.bumpMapTransform),m.bumpScale.value=u.bumpScale,u.side===pe&&(m.bumpScale.value*=-1)),u.normalMap&&(m.normalMap.value=u.normalMap,e(u.normalMap,m.normalMapTransform),m.normalScale.value.copy(u.normalScale),u.side===pe&&m.normalScale.value.negate()),u.displacementMap&&(m.displacementMap.value=u.displacementMap,e(u.displacementMap,m.displacementMapTransform),m.displacementScale.value=u.displacementScale,m.displacementBias.value=u.displacementBias),u.emissiveMap&&(m.emissiveMap.value=u.emissiveMap,e(u.emissiveMap,m.emissiveMapTransform)),u.specularMap&&(m.specularMap.value=u.specularMap,e(u.specularMap,m.specularMapTransform)),u.alphaTest>0&&(m.alphaTest.value=u.alphaTest);const E=t.get(u).envMap;if(E&&(m.envMap.value=E,m.flipEnvMap.value=E.isCubeTexture&&E.isRenderTargetTexture===!1?-1:1,m.reflectivity.value=u.reflectivity,m.ior.value=u.ior,m.refractionRatio.value=u.refractionRatio),u.lightMap){m.lightMap.value=u.lightMap;const x=a.useLegacyLights===!0?Math.PI:1;m.lightMapIntensity.value=u.lightMapIntensity*x,e(u.lightMap,m.lightMapTransform)}u.aoMap&&(m.aoMap.value=u.aoMap,m.aoMapIntensity.value=u.aoMapIntensity,e(u.aoMap,m.aoMapTransform))}function o(m,u){m.diffuse.value.copy(u.color),m.opacity.value=u.opacity,u.map&&(m.map.value=u.map,e(u.map,m.mapTransform))}function s(m,u){m.dashSize.value=u.dashSize,m.totalSize.value=u.dashSize+u.gapSize,m.scale.value=u.scale}function c(m,u,E,x){m.diffuse.value.copy(u.color),m.opacity.value=u.opacity,m.size.value=u.size*E,m.scale.value=x*.5,u.map&&(m.map.value=u.map,e(u.map,m.uvTransform)),u.alphaMap&&(m.alphaMap.value=u.alphaMap),u.alphaTest>0&&(m.alphaTest.value=u.alphaTest)}function l(m,u){m.diffuse.value.copy(u.color),m.opacity.value=u.opacity,m.rotation.value=u.rotation,u.map&&(m.map.value=u.map,e(u.map,m.mapTransform)),u.alphaMap&&(m.alphaMap.value=u.alphaMap),u.alphaTest>0&&(m.alphaTest.value=u.alphaTest)}function h(m,u){m.specular.value.copy(u.specular),m.shininess.value=Math.max(u.shininess,1e-4)}function d(m,u){u.gradientMap&&(m.gradientMap.value=u.gradientMap)}function f(m,u){m.metalness.value=u.metalness,u.metalnessMap&&(m.metalnessMap.value=u.metalnessMap,e(u.metalnessMap,m.metalnessMapTransform)),m.roughness.value=u.roughness,u.roughnessMap&&(m.roughnessMap.value=u.roughnessMap,e(u.roughnessMap,m.roughnessMapTransform)),t.get(u).envMap&&(m.envMapIntensity.value=u.envMapIntensity)}function p(m,u,E){m.ior.value=u.ior,u.sheen>0&&(m.sheenColor.value.copy(u.sheenColor).multiplyScalar(u.sheen),m.sheenRoughness.value=u.sheenRoughness,u.sheenColorMap&&(m.sheenColorMap.value=u.sheenColorMap,e(u.sheenColorMap,m.sheenColorMapTransform)),u.sheenRoughnessMap&&(m.sheenRoughnessMap.value=u.sheenRoughnessMap,e(u.sheenRoughnessMap,m.sheenRoughnessMapTransform))),u.clearcoat>0&&(m.clearcoat.value=u.clearcoat,m.clearcoatRoughness.value=u.clearcoatRoughness,u.clearcoatMap&&(m.clearcoatMap.value=u.clearcoatMap,e(u.clearcoatMap,m.clearcoatMapTransform)),u.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=u.clearcoatRoughnessMap,e(u.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),u.clearcoatNormalMap&&(m.clearcoatNormalMap.value=u.clearcoatNormalMap,e(u.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(u.clearcoatNormalScale),u.side===pe&&m.clearcoatNormalScale.value.negate())),u.iridescence>0&&(m.iridescence.value=u.iridescence,m.iridescenceIOR.value=u.iridescenceIOR,m.iridescenceThicknessMinimum.value=u.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=u.iridescenceThicknessRange[1],u.iridescenceMap&&(m.iridescenceMap.value=u.iridescenceMap,e(u.iridescenceMap,m.iridescenceMapTransform)),u.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=u.iridescenceThicknessMap,e(u.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),u.transmission>0&&(m.transmission.value=u.transmission,m.transmissionSamplerMap.value=E.texture,m.transmissionSamplerSize.value.set(E.width,E.height),u.transmissionMap&&(m.transmissionMap.value=u.transmissionMap,e(u.transmissionMap,m.transmissionMapTransform)),m.thickness.value=u.thickness,u.thicknessMap&&(m.thicknessMap.value=u.thicknessMap,e(u.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=u.attenuationDistance,m.attenuationColor.value.copy(u.attenuationColor)),m.specularIntensity.value=u.specularIntensity,m.specularColor.value.copy(u.specularColor),u.specularColorMap&&(m.specularColorMap.value=u.specularColorMap,e(u.specularColorMap,m.specularColorMapTransform)),u.specularIntensityMap&&(m.specularIntensityMap.value=u.specularIntensityMap,e(u.specularIntensityMap,m.specularIntensityMapTransform))}function g(m,u){u.matcap&&(m.matcap.value=u.matcap)}function _(m,u){const E=t.get(u).light;m.referencePosition.value.setFromMatrixPosition(E.matrixWorld),m.nearDistance.value=E.shadow.camera.near,m.farDistance.value=E.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:i}}function Kd(a,t,e,n){let i={},r={},o=[];const s=e.isWebGL2?a.getParameter(35375):0;function c(E,x){const b=x.program;n.uniformBlockBinding(E,b)}function l(E,x){let b=i[E.id];b===void 0&&(g(E),b=h(E),i[E.id]=b,E.addEventListener("dispose",m));const y=x.program;n.updateUBOMapping(E,y);const C=t.render.frame;r[E.id]!==C&&(f(E),r[E.id]=C)}function h(E){const x=d();E.__bindingPointIndex=x;const b=a.createBuffer(),y=E.__size,C=E.usage;return a.bindBuffer(35345,b),a.bufferData(35345,y,C),a.bindBuffer(35345,null),a.bindBufferBase(35345,x,b),b}function d(){for(let E=0;E<s;E++)if(o.indexOf(E)===-1)return o.push(E),E;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function f(E){const x=i[E.id],b=E.uniforms,y=E.__cache;a.bindBuffer(35345,x);for(let C=0,D=b.length;C<D;C++){const R=b[C];if(p(R,C,y)===!0){const M=R.__offset,T=Array.isArray(R.value)?R.value:[R.value];let X=0;for(let q=0;q<T.length;q++){const I=T[q],N=_(I);typeof I=="number"?(R.__data[0]=I,a.bufferSubData(35345,M+X,R.__data)):I.isMatrix3?(R.__data[0]=I.elements[0],R.__data[1]=I.elements[1],R.__data[2]=I.elements[2],R.__data[3]=I.elements[0],R.__data[4]=I.elements[3],R.__data[5]=I.elements[4],R.__data[6]=I.elements[5],R.__data[7]=I.elements[0],R.__data[8]=I.elements[6],R.__data[9]=I.elements[7],R.__data[10]=I.elements[8],R.__data[11]=I.elements[0]):(I.toArray(R.__data,X),X+=N.storage/Float32Array.BYTES_PER_ELEMENT)}a.bufferSubData(35345,M,R.__data)}}a.bindBuffer(35345,null)}function p(E,x,b){const y=E.value;if(b[x]===void 0){if(typeof y=="number")b[x]=y;else{const C=Array.isArray(y)?y:[y],D=[];for(let R=0;R<C.length;R++)D.push(C[R].clone());b[x]=D}return!0}else if(typeof y=="number"){if(b[x]!==y)return b[x]=y,!0}else{const C=Array.isArray(b[x])?b[x]:[b[x]],D=Array.isArray(y)?y:[y];for(let R=0;R<C.length;R++){const M=C[R];if(M.equals(D[R])===!1)return M.copy(D[R]),!0}}return!1}function g(E){const x=E.uniforms;let b=0;const y=16;let C=0;for(let D=0,R=x.length;D<R;D++){const M=x[D],T={boundary:0,storage:0},X=Array.isArray(M.value)?M.value:[M.value];for(let q=0,I=X.length;q<I;q++){const N=X[q],B=_(N);T.boundary+=B.boundary,T.storage+=B.storage}if(M.__data=new Float32Array(T.storage/Float32Array.BYTES_PER_ELEMENT),M.__offset=b,D>0){C=b%y;const q=y-C;C!==0&&q-T.boundary<0&&(b+=y-C,M.__offset=b)}b+=T.storage}return C=b%y,C>0&&(b+=y-C),E.__size=b,E.__cache={},this}function _(E){const x={boundary:0,storage:0};return typeof E=="number"?(x.boundary=4,x.storage=4):E.isVector2?(x.boundary=8,x.storage=8):E.isVector3||E.isColor?(x.boundary=16,x.storage=12):E.isVector4?(x.boundary=16,x.storage=16):E.isMatrix3?(x.boundary=48,x.storage=48):E.isMatrix4?(x.boundary=64,x.storage=64):E.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",E),x}function m(E){const x=E.target;x.removeEventListener("dispose",m);const b=o.indexOf(x.__bindingPointIndex);o.splice(b,1),a.deleteBuffer(i[x.id]),delete i[x.id],delete r[x.id]}function u(){for(const E in i)a.deleteBuffer(i[E]);o=[],i={},r={}}return{bind:c,update:l,dispose:u}}function $d(){const a=Ni("canvas");return a.style.display="block",a}class Ya{constructor(t={}){const{canvas:e=$d(),context:n=null,depth:i=!0,stencil:r=!0,alpha:o=!1,antialias:s=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:d=!1}=t;this.isWebGLRenderer=!0;let f;n!==null?f=n.getContextAttributes().alpha:f=o;let p=null,g=null;const _=[],m=[];this.domElement=e,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.outputEncoding=gn,this.useLegacyLights=!0,this.toneMapping=je,this.toneMappingExposure=1;const u=this;let E=!1,x=0,b=0,y=null,C=-1,D=null;const R=new ie,M=new ie;let T=null,X=e.width,q=e.height,I=1,N=null,B=null;const Q=new ie(0,0,X,q),et=new ie(0,0,X,q);let j=!1;const Z=new Fr;let nt=!1,vt=!1,ct=null;const V=new Kt,J=new P,st={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};function lt(){return y===null?I:1}let z=n;function bt(S,O){for(let k=0;k<S.length;k++){const U=S[k],W=e.getContext(U,O);if(W!==null)return W}return null}try{const S={alpha:!0,depth:i,stencil:r,antialias:s,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:d};if("setAttribute"in e&&e.setAttribute("data-engine","three.js r151"),e.addEventListener("webglcontextlost",pt,!1),e.addEventListener("webglcontextrestored",wt,!1),e.addEventListener("webglcontextcreationerror",Lt,!1),z===null){const O=["webgl2","webgl","experimental-webgl"];if(u.isWebGL1Renderer===!0&&O.shift(),z=bt(O,S),z===null)throw bt(O)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}z.getShaderPrecisionFormat===void 0&&(z.getShaderPrecisionFormat=function(){return{rangeMin:1,rangeMax:1,precision:1}})}catch(S){throw console.error("THREE.WebGLRenderer: "+S.message),S}let St,it,Mt,Ft,gt,Tt,Wt,kt,jt,qt,Nt,Bt,ae,w,v,G,K,tt,at,A,Y,F,ot,ft;function mt(){St=new cu(z),it=new iu(z,St,t),St.init(it),F=new qd(z,St,it),Mt=new Hd(z,St,it),Ft=new du,gt=new Ld,Tt=new Wd(z,St,Mt,gt,it,F,Ft),Wt=new su(u),kt=new lu(u),jt=new wl(z,it),ot=new eu(z,St,jt,it),qt=new hu(z,jt,Ft,ot),Nt=new gu(z,qt,jt,Ft),at=new mu(z,it,Tt),G=new ru(gt),Bt=new Cd(u,Wt,kt,St,it,ot,G),ae=new Zd(u,gt),w=new Rd,v=new Fd(St,it),tt=new tu(u,Wt,kt,Mt,Nt,f,c),K=new kd(u,Nt,it),ft=new Kd(z,Ft,it,Mt),A=new nu(z,St,Ft,it),Y=new uu(z,St,Ft,it),Ft.programs=Bt.programs,u.capabilities=it,u.extensions=St,u.properties=gt,u.renderLists=w,u.shadowMap=K,u.state=Mt,u.info=Ft}mt();const ht=new Jd(u,z);this.xr=ht,this.getContext=function(){return z},this.getContextAttributes=function(){return z.getContextAttributes()},this.forceContextLoss=function(){const S=St.get("WEBGL_lose_context");S&&S.loseContext()},this.forceContextRestore=function(){const S=St.get("WEBGL_lose_context");S&&S.restoreContext()},this.getPixelRatio=function(){return I},this.setPixelRatio=function(S){S!==void 0&&(I=S,this.setSize(X,q,!1))},this.getSize=function(S){return S.set(X,q)},this.setSize=function(S,O,k=!0){if(ht.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}X=S,q=O,e.width=Math.floor(S*I),e.height=Math.floor(O*I),k===!0&&(e.style.width=S+"px",e.style.height=O+"px"),this.setViewport(0,0,S,O)},this.getDrawingBufferSize=function(S){return S.set(X*I,q*I).floor()},this.setDrawingBufferSize=function(S,O,k){X=S,q=O,I=k,e.width=Math.floor(S*k),e.height=Math.floor(O*k),this.setViewport(0,0,S,O)},this.getCurrentViewport=function(S){return S.copy(R)},this.getViewport=function(S){return S.copy(Q)},this.setViewport=function(S,O,k,U){S.isVector4?Q.set(S.x,S.y,S.z,S.w):Q.set(S,O,k,U),Mt.viewport(R.copy(Q).multiplyScalar(I).floor())},this.getScissor=function(S){return S.copy(et)},this.setScissor=function(S,O,k,U){S.isVector4?et.set(S.x,S.y,S.z,S.w):et.set(S,O,k,U),Mt.scissor(M.copy(et).multiplyScalar(I).floor())},this.getScissorTest=function(){return j},this.setScissorTest=function(S){Mt.setScissorTest(j=S)},this.setOpaqueSort=function(S){N=S},this.setTransparentSort=function(S){B=S},this.getClearColor=function(S){return S.copy(tt.getClearColor())},this.setClearColor=function(){tt.setClearColor.apply(tt,arguments)},this.getClearAlpha=function(){return tt.getClearAlpha()},this.setClearAlpha=function(){tt.setClearAlpha.apply(tt,arguments)},this.clear=function(S=!0,O=!0,k=!0){let U=0;S&&(U|=16384),O&&(U|=256),k&&(U|=1024),z.clear(U)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",pt,!1),e.removeEventListener("webglcontextrestored",wt,!1),e.removeEventListener("webglcontextcreationerror",Lt,!1),w.dispose(),v.dispose(),gt.dispose(),Wt.dispose(),kt.dispose(),Nt.dispose(),ot.dispose(),ft.dispose(),Bt.dispose(),ht.dispose(),ht.removeEventListener("sessionstart",_t),ht.removeEventListener("sessionend",Vt),ct&&(ct.dispose(),ct=null),Ht.stop()};function pt(S){S.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),E=!0}function wt(){console.log("THREE.WebGLRenderer: Context Restored."),E=!1;const S=Ft.autoReset,O=K.enabled,k=K.autoUpdate,U=K.needsUpdate,W=K.type;mt(),Ft.autoReset=S,K.enabled=O,K.autoUpdate=k,K.needsUpdate=U,K.type=W}function Lt(S){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",S.statusMessage)}function Xt(S){const O=S.target;O.removeEventListener("dispose",Xt),L(O)}function L(S){H(S),gt.remove(S)}function H(S){const O=gt.get(S).programs;O!==void 0&&(O.forEach(function(k){Bt.releaseProgram(k)}),S.isShaderMaterial&&Bt.releaseShaderCache(S))}this.renderBufferDirect=function(S,O,k,U,W,xt){O===null&&(O=st);const yt=W.isMesh&&W.matrixWorld.determinant()<0,Et=$a(S,O,k,U,W);Mt.setMaterial(U,yt);let At=k.index,Pt=1;U.wireframe===!0&&(At=qt.getWireframeAttribute(k),Pt=2);const Rt=k.drawRange,Dt=k.attributes.position;let zt=Rt.start*Pt,he=(Rt.start+Rt.count)*Pt;xt!==null&&(zt=Math.max(zt,xt.start*Pt),he=Math.min(he,(xt.start+xt.count)*Pt)),At!==null?(zt=Math.max(zt,0),he=Math.min(he,At.count)):Dt!=null&&(zt=Math.max(zt,0),he=Math.min(he,Dt.count));const Te=he-zt;if(Te<0||Te===1/0)return;ot.setup(W,U,Et,k,At);let nn,Jt=A;if(At!==null&&(nn=jt.get(At),Jt=Y,Jt.setIndex(nn)),W.isMesh)U.wireframe===!0?(Mt.setLineWidth(U.wireframeLinewidth*lt()),Jt.setMode(1)):Jt.setMode(4);else if(W.isLine){let Ut=U.linewidth;Ut===void 0&&(Ut=1),Mt.setLineWidth(Ut*lt()),W.isLineSegments?Jt.setMode(1):W.isLineLoop?Jt.setMode(2):Jt.setMode(3)}else W.isPoints?Jt.setMode(0):W.isSprite&&Jt.setMode(4);if(W.isInstancedMesh)Jt.renderInstances(zt,Te,W.count);else if(k.isInstancedBufferGeometry){const Ut=k._maxInstanceCount!==void 0?k._maxInstanceCount:1/0,Gi=Math.min(k.instanceCount,Ut);Jt.renderInstances(zt,Te,Gi)}else Jt.render(zt,Te)},this.compile=function(S,O){function k(U,W,xt){U.transparent===!0&&U.side===Xe&&U.forceSinglePass===!1?(U.side=pe,U.needsUpdate=!0,ui(U,W,xt),U.side=en,U.needsUpdate=!0,ui(U,W,xt),U.side=Xe):ui(U,W,xt)}g=v.get(S),g.init(),m.push(g),S.traverseVisible(function(U){U.isLight&&U.layers.test(O.layers)&&(g.pushLight(U),U.castShadow&&g.pushShadow(U))}),g.setupLights(u.useLegacyLights),S.traverse(function(U){const W=U.material;if(W)if(Array.isArray(W))for(let xt=0;xt<W.length;xt++){const yt=W[xt];k(yt,S,U)}else k(W,S,U)}),m.pop(),g=null};let $=null;function ut(S){$&&$(S)}function _t(){Ht.stop()}function Vt(){Ht.start()}const Ht=new Va;Ht.setAnimationLoop(ut),typeof self<"u"&&Ht.setContext(self),this.setAnimationLoop=function(S){$=S,ht.setAnimationLoop(S),S===null?Ht.stop():Ht.start()},ht.addEventListener("sessionstart",_t),ht.addEventListener("sessionend",Vt),this.render=function(S,O){if(O!==void 0&&O.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(E===!0)return;S.matrixWorldAutoUpdate===!0&&S.updateMatrixWorld(),O.parent===null&&O.matrixWorldAutoUpdate===!0&&O.updateMatrixWorld(),ht.enabled===!0&&ht.isPresenting===!0&&(ht.cameraAutoUpdate===!0&&ht.updateCamera(O),O=ht.getCamera()),S.isScene===!0&&S.onBeforeRender(u,S,O,y),g=v.get(S,m.length),g.init(),m.push(g),V.multiplyMatrices(O.projectionMatrix,O.matrixWorldInverse),Z.setFromProjectionMatrix(V),vt=this.localClippingEnabled,nt=G.init(this.clippingPlanes,vt),p=w.get(S,_.length),p.init(),_.push(p),oe(S,O,0,u.sortObjects),p.finish(),u.sortObjects===!0&&p.sort(N,B),nt===!0&&G.beginShadows();const k=g.state.shadowsArray;if(K.render(k,S,O),nt===!0&&G.endShadows(),this.info.autoReset===!0&&this.info.reset(),tt.render(p,S),g.setupLights(u.useLegacyLights),O.isArrayCamera){const U=O.cameras;for(let W=0,xt=U.length;W<xt;W++){const yt=U[W];Je(p,S,yt,yt.viewport)}}else Je(p,S,O);y!==null&&(Tt.updateMultisampleRenderTarget(y),Tt.updateRenderTargetMipmap(y)),S.isScene===!0&&S.onAfterRender(u,S,O),ot.resetDefaultState(),C=-1,D=null,m.pop(),m.length>0?g=m[m.length-1]:g=null,_.pop(),_.length>0?p=_[_.length-1]:p=null};function oe(S,O,k,U){if(S.visible===!1)return;if(S.layers.test(O.layers)){if(S.isGroup)k=S.renderOrder;else if(S.isLOD)S.autoUpdate===!0&&S.update(O);else if(S.isLight)g.pushLight(S),S.castShadow&&g.pushShadow(S);else if(S.isSprite){if(!S.frustumCulled||Z.intersectsSprite(S)){U&&J.setFromMatrixPosition(S.matrixWorld).applyMatrix4(V);const yt=Nt.update(S),Et=S.material;Et.visible&&p.push(S,yt,Et,k,J.z,null)}}else if((S.isMesh||S.isLine||S.isPoints)&&(S.isSkinnedMesh&&S.skeleton.frame!==Ft.render.frame&&(S.skeleton.update(),S.skeleton.frame=Ft.render.frame),!S.frustumCulled||Z.intersectsObject(S))){U&&J.setFromMatrixPosition(S.matrixWorld).applyMatrix4(V);const yt=Nt.update(S),Et=S.material;if(Array.isArray(Et)){const At=yt.groups;for(let Pt=0,Rt=At.length;Pt<Rt;Pt++){const Dt=At[Pt],zt=Et[Dt.materialIndex];zt&&zt.visible&&p.push(S,yt,zt,k,J.z,Dt)}}else Et.visible&&p.push(S,yt,Et,k,J.z,null)}}const xt=S.children;for(let yt=0,Et=xt.length;yt<Et;yt++)oe(xt[yt],O,k,U)}function Je(S,O,k,U){const W=S.opaque,xt=S.transmissive,yt=S.transparent;g.setupLightsView(k),nt===!0&&G.setGlobalState(u.clippingPlanes,k),xt.length>0&&Yt(W,xt,O,k),U&&Mt.viewport(R.copy(U)),W.length>0&&Se(W,O,k),xt.length>0&&Se(xt,O,k),yt.length>0&&Se(yt,O,k),Mt.buffers.depth.setTest(!0),Mt.buffers.depth.setMask(!0),Mt.buffers.color.setMask(!0),Mt.setPolygonOffset(!1)}function Yt(S,O,k,U){if(ct===null){const Et=it.isWebGL2;ct=new _n(1024,1024,{generateMipmaps:!0,type:St.has("EXT_color_buffer_half_float")?ai:mn,minFilter:si,samples:Et&&s===!0?4:0})}const W=u.getRenderTarget();u.setRenderTarget(ct),u.clear();const xt=u.toneMapping;u.toneMapping=je,Se(S,k,U),Tt.updateMultisampleRenderTarget(ct),Tt.updateRenderTargetMipmap(ct);let yt=!1;for(let Et=0,At=O.length;Et<At;Et++){const Pt=O[Et],Rt=Pt.object,Dt=Pt.geometry,zt=Pt.material,he=Pt.group;if(zt.side===Xe&&Rt.layers.test(U.layers)){const Te=zt.side;zt.side=pe,zt.needsUpdate=!0,Ue(Rt,k,U,Dt,zt,he),zt.side=Te,zt.needsUpdate=!0,yt=!0}}yt===!0&&(Tt.updateMultisampleRenderTarget(ct),Tt.updateRenderTargetMipmap(ct)),u.setRenderTarget(W),u.toneMapping=xt}function Se(S,O,k){const U=O.isScene===!0?O.overrideMaterial:null;for(let W=0,xt=S.length;W<xt;W++){const yt=S[W],Et=yt.object,At=yt.geometry,Pt=U===null?yt.material:U,Rt=yt.group;Et.layers.test(k.layers)&&Ue(Et,O,k,At,Pt,Rt)}}function Ue(S,O,k,U,W,xt){S.onBeforeRender(u,O,k,U,W,xt),S.modelViewMatrix.multiplyMatrices(k.matrixWorldInverse,S.matrixWorld),S.normalMatrix.getNormalMatrix(S.modelViewMatrix),W.onBeforeRender(u,O,k,U,S,xt),W.transparent===!0&&W.side===Xe&&W.forceSinglePass===!1?(W.side=pe,W.needsUpdate=!0,u.renderBufferDirect(k,O,U,W,S,xt),W.side=en,W.needsUpdate=!0,u.renderBufferDirect(k,O,U,W,S,xt),W.side=Xe):u.renderBufferDirect(k,O,U,W,S,xt),S.onAfterRender(u,O,k,U,W,xt)}function ui(S,O,k){O.isScene!==!0&&(O=st);const U=gt.get(S),W=g.state.lights,xt=g.state.shadowsArray,yt=W.state.version,Et=Bt.getParameters(S,W.state,xt,O,k),At=Bt.getProgramCacheKey(Et);let Pt=U.programs;U.environment=S.isMeshStandardMaterial?O.environment:null,U.fog=O.fog,U.envMap=(S.isMeshStandardMaterial?kt:Wt).get(S.envMap||U.environment),Pt===void 0&&(S.addEventListener("dispose",Xt),Pt=new Map,U.programs=Pt);let Rt=Pt.get(At);if(Rt!==void 0){if(U.currentProgram===Rt&&U.lightsStateVersion===yt)return Yr(S,Et),Rt}else Et.uniforms=Bt.getUniforms(S),S.onBuild(k,Et,u),S.onBeforeCompile(Et,u),Rt=Bt.acquireProgram(Et,At),Pt.set(At,Rt),U.uniforms=Et.uniforms;const Dt=U.uniforms;(!S.isShaderMaterial&&!S.isRawShaderMaterial||S.clipping===!0)&&(Dt.clippingPlanes=G.uniform),Yr(S,Et),U.needsLights=to(S),U.lightsStateVersion=yt,U.needsLights&&(Dt.ambientLightColor.value=W.state.ambient,Dt.lightProbe.value=W.state.probe,Dt.directionalLights.value=W.state.directional,Dt.directionalLightShadows.value=W.state.directionalShadow,Dt.spotLights.value=W.state.spot,Dt.spotLightShadows.value=W.state.spotShadow,Dt.rectAreaLights.value=W.state.rectArea,Dt.ltc_1.value=W.state.rectAreaLTC1,Dt.ltc_2.value=W.state.rectAreaLTC2,Dt.pointLights.value=W.state.point,Dt.pointLightShadows.value=W.state.pointShadow,Dt.hemisphereLights.value=W.state.hemi,Dt.directionalShadowMap.value=W.state.directionalShadowMap,Dt.directionalShadowMatrix.value=W.state.directionalShadowMatrix,Dt.spotShadowMap.value=W.state.spotShadowMap,Dt.spotLightMatrix.value=W.state.spotLightMatrix,Dt.spotLightMap.value=W.state.spotLightMap,Dt.pointShadowMap.value=W.state.pointShadowMap,Dt.pointShadowMatrix.value=W.state.pointShadowMatrix);const zt=Rt.getUniforms(),he=Ui.seqWithValue(zt.seq,Dt);return U.currentProgram=Rt,U.uniformsList=he,Rt}function Yr(S,O){const k=gt.get(S);k.outputEncoding=O.outputEncoding,k.instancing=O.instancing,k.skinning=O.skinning,k.morphTargets=O.morphTargets,k.morphNormals=O.morphNormals,k.morphColors=O.morphColors,k.morphTargetsCount=O.morphTargetsCount,k.numClippingPlanes=O.numClippingPlanes,k.numIntersection=O.numClipIntersection,k.vertexAlphas=O.vertexAlphas,k.vertexTangents=O.vertexTangents,k.toneMapping=O.toneMapping}function $a(S,O,k,U,W){O.isScene!==!0&&(O=st),Tt.resetTextureUnits();const xt=O.fog,yt=U.isMeshStandardMaterial?O.environment:null,Et=y===null?u.outputEncoding:y.isXRRenderTarget===!0?y.texture.encoding:gn,At=(U.isMeshStandardMaterial?kt:Wt).get(U.envMap||yt),Pt=U.vertexColors===!0&&!!k.attributes.color&&k.attributes.color.itemSize===4,Rt=!!U.normalMap&&!!k.attributes.tangent,Dt=!!k.morphAttributes.position,zt=!!k.morphAttributes.normal,he=!!k.morphAttributes.color,Te=U.toneMapped?u.toneMapping:je,nn=k.morphAttributes.position||k.morphAttributes.normal||k.morphAttributes.color,Jt=nn!==void 0?nn.length:0,Ut=gt.get(U),Gi=g.state.lights;if(nt===!0&&(vt===!0||S!==D)){const me=S===D&&U.id===C;G.setState(U,S,me)}let te=!1;U.version===Ut.__version?(Ut.needsLights&&Ut.lightsStateVersion!==Gi.state.version||Ut.outputEncoding!==Et||W.isInstancedMesh&&Ut.instancing===!1||!W.isInstancedMesh&&Ut.instancing===!0||W.isSkinnedMesh&&Ut.skinning===!1||!W.isSkinnedMesh&&Ut.skinning===!0||Ut.envMap!==At||U.fog===!0&&Ut.fog!==xt||Ut.numClippingPlanes!==void 0&&(Ut.numClippingPlanes!==G.numPlanes||Ut.numIntersection!==G.numIntersection)||Ut.vertexAlphas!==Pt||Ut.vertexTangents!==Rt||Ut.morphTargets!==Dt||Ut.morphNormals!==zt||Ut.morphColors!==he||Ut.toneMapping!==Te||it.isWebGL2===!0&&Ut.morphTargetsCount!==Jt)&&(te=!0):(te=!0,Ut.__version=U.version);let rn=Ut.currentProgram;te===!0&&(rn=ui(U,O,W));let jr=!1,Zn=!1,Vi=!1;const ue=rn.getUniforms(),sn=Ut.uniforms;if(Mt.useProgram(rn.program)&&(jr=!0,Zn=!0,Vi=!0),U.id!==C&&(C=U.id,Zn=!0),jr||D!==S){if(ue.setValue(z,"projectionMatrix",S.projectionMatrix),it.logarithmicDepthBuffer&&ue.setValue(z,"logDepthBufFC",2/(Math.log(S.far+1)/Math.LN2)),D!==S&&(D=S,Zn=!0,Vi=!0),U.isShaderMaterial||U.isMeshPhongMaterial||U.isMeshToonMaterial||U.isMeshStandardMaterial||U.envMap){const me=ue.map.cameraPosition;me!==void 0&&me.setValue(z,J.setFromMatrixPosition(S.matrixWorld))}(U.isMeshPhongMaterial||U.isMeshToonMaterial||U.isMeshLambertMaterial||U.isMeshBasicMaterial||U.isMeshStandardMaterial||U.isShaderMaterial)&&ue.setValue(z,"isOrthographic",S.isOrthographicCamera===!0),(U.isMeshPhongMaterial||U.isMeshToonMaterial||U.isMeshLambertMaterial||U.isMeshBasicMaterial||U.isMeshStandardMaterial||U.isShaderMaterial||U.isShadowMaterial||W.isSkinnedMesh)&&ue.setValue(z,"viewMatrix",S.matrixWorldInverse)}if(W.isSkinnedMesh){ue.setOptional(z,W,"bindMatrix"),ue.setOptional(z,W,"bindMatrixInverse");const me=W.skeleton;me&&(it.floatVertexTextures?(me.boneTexture===null&&me.computeBoneTexture(),ue.setValue(z,"boneTexture",me.boneTexture,Tt),ue.setValue(z,"boneTextureSize",me.boneTextureSize)):console.warn("THREE.WebGLRenderer: SkinnedMesh can only be used with WebGL 2. With WebGL 1 OES_texture_float and vertex textures support is required."))}const ki=k.morphAttributes;if((ki.position!==void 0||ki.normal!==void 0||ki.color!==void 0&&it.isWebGL2===!0)&&at.update(W,k,rn),(Zn||Ut.receiveShadow!==W.receiveShadow)&&(Ut.receiveShadow=W.receiveShadow,ue.setValue(z,"receiveShadow",W.receiveShadow)),U.isMeshGouraudMaterial&&U.envMap!==null&&(sn.envMap.value=At,sn.flipEnvMap.value=At.isCubeTexture&&At.isRenderTargetTexture===!1?-1:1),Zn&&(ue.setValue(z,"toneMappingExposure",u.toneMappingExposure),Ut.needsLights&&Qa(sn,Vi),xt&&U.fog===!0&&ae.refreshFogUniforms(sn,xt),ae.refreshMaterialUniforms(sn,U,I,q,ct),Ui.upload(z,Ut.uniformsList,sn,Tt)),U.isShaderMaterial&&U.uniformsNeedUpdate===!0&&(Ui.upload(z,Ut.uniformsList,sn,Tt),U.uniformsNeedUpdate=!1),U.isSpriteMaterial&&ue.setValue(z,"center",W.center),ue.setValue(z,"modelViewMatrix",W.modelViewMatrix),ue.setValue(z,"normalMatrix",W.normalMatrix),ue.setValue(z,"modelMatrix",W.matrixWorld),U.isShaderMaterial||U.isRawShaderMaterial){const me=U.uniformsGroups;for(let Hi=0,eo=me.length;Hi<eo;Hi++)if(it.isWebGL2){const Jr=me[Hi];ft.update(Jr,rn),ft.bind(Jr,rn)}else console.warn("THREE.WebGLRenderer: Uniform Buffer Objects can only be used with WebGL 2.")}return rn}function Qa(S,O){S.ambientLightColor.needsUpdate=O,S.lightProbe.needsUpdate=O,S.directionalLights.needsUpdate=O,S.directionalLightShadows.needsUpdate=O,S.pointLights.needsUpdate=O,S.pointLightShadows.needsUpdate=O,S.spotLights.needsUpdate=O,S.spotLightShadows.needsUpdate=O,S.rectAreaLights.needsUpdate=O,S.hemisphereLights.needsUpdate=O}function to(S){return S.isMeshLambertMaterial||S.isMeshToonMaterial||S.isMeshPhongMaterial||S.isMeshStandardMaterial||S.isShadowMaterial||S.isShaderMaterial&&S.lights===!0}this.getActiveCubeFace=function(){return x},this.getActiveMipmapLevel=function(){return b},this.getRenderTarget=function(){return y},this.setRenderTargetTextures=function(S,O,k){gt.get(S.texture).__webglTexture=O,gt.get(S.depthTexture).__webglTexture=k;const U=gt.get(S);U.__hasExternalTextures=!0,U.__hasExternalTextures&&(U.__autoAllocateDepthBuffer=k===void 0,U.__autoAllocateDepthBuffer||St.has("WEBGL_multisampled_render_to_texture")===!0&&(console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"),U.__useRenderToTexture=!1))},this.setRenderTargetFramebuffer=function(S,O){const k=gt.get(S);k.__webglFramebuffer=O,k.__useDefaultFramebuffer=O===void 0},this.setRenderTarget=function(S,O=0,k=0){y=S,x=O,b=k;let U=!0,W=null,xt=!1,yt=!1;if(S){const At=gt.get(S);At.__useDefaultFramebuffer!==void 0?(Mt.bindFramebuffer(36160,null),U=!1):At.__webglFramebuffer===void 0?Tt.setupRenderTarget(S):At.__hasExternalTextures&&Tt.rebindTextures(S,gt.get(S.texture).__webglTexture,gt.get(S.depthTexture).__webglTexture);const Pt=S.texture;(Pt.isData3DTexture||Pt.isDataArrayTexture||Pt.isCompressedArrayTexture)&&(yt=!0);const Rt=gt.get(S).__webglFramebuffer;S.isWebGLCubeRenderTarget?(W=Rt[O],xt=!0):it.isWebGL2&&S.samples>0&&Tt.useMultisampledRTT(S)===!1?W=gt.get(S).__webglMultisampledFramebuffer:W=Rt,R.copy(S.viewport),M.copy(S.scissor),T=S.scissorTest}else R.copy(Q).multiplyScalar(I).floor(),M.copy(et).multiplyScalar(I).floor(),T=j;if(Mt.bindFramebuffer(36160,W)&&it.drawBuffers&&U&&Mt.drawBuffers(S,W),Mt.viewport(R),Mt.scissor(M),Mt.setScissorTest(T),xt){const At=gt.get(S.texture);z.framebufferTexture2D(36160,36064,34069+O,At.__webglTexture,k)}else if(yt){const At=gt.get(S.texture),Pt=O||0;z.framebufferTextureLayer(36160,36064,At.__webglTexture,k||0,Pt)}C=-1},this.readRenderTargetPixels=function(S,O,k,U,W,xt,yt){if(!(S&&S.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Et=gt.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&yt!==void 0&&(Et=Et[yt]),Et){Mt.bindFramebuffer(36160,Et);try{const At=S.texture,Pt=At.format,Rt=At.type;if(Pt!==De&&F.convert(Pt)!==z.getParameter(35739)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}const Dt=Rt===ai&&(St.has("EXT_color_buffer_half_float")||it.isWebGL2&&St.has("EXT_color_buffer_float"));if(Rt!==mn&&F.convert(Rt)!==z.getParameter(35738)&&!(Rt===fn&&(it.isWebGL2||St.has("OES_texture_float")||St.has("WEBGL_color_buffer_float")))&&!Dt){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}O>=0&&O<=S.width-U&&k>=0&&k<=S.height-W&&z.readPixels(O,k,U,W,F.convert(Pt),F.convert(Rt),xt)}finally{const At=y!==null?gt.get(y).__webglFramebuffer:null;Mt.bindFramebuffer(36160,At)}}},this.copyFramebufferToTexture=function(S,O,k=0){const U=Math.pow(2,-k),W=Math.floor(O.image.width*U),xt=Math.floor(O.image.height*U);Tt.setTexture2D(O,0),z.copyTexSubImage2D(3553,k,0,0,S.x,S.y,W,xt),Mt.unbindTexture()},this.copyTextureToTexture=function(S,O,k,U=0){const W=O.image.width,xt=O.image.height,yt=F.convert(k.format),Et=F.convert(k.type);Tt.setTexture2D(k,0),z.pixelStorei(37440,k.flipY),z.pixelStorei(37441,k.premultiplyAlpha),z.pixelStorei(3317,k.unpackAlignment),O.isDataTexture?z.texSubImage2D(3553,U,S.x,S.y,W,xt,yt,Et,O.image.data):O.isCompressedTexture?z.compressedTexSubImage2D(3553,U,S.x,S.y,O.mipmaps[0].width,O.mipmaps[0].height,yt,O.mipmaps[0].data):z.texSubImage2D(3553,U,S.x,S.y,yt,Et,O.image),U===0&&k.generateMipmaps&&z.generateMipmap(3553),Mt.unbindTexture()},this.copyTextureToTexture3D=function(S,O,k,U,W=0){if(u.isWebGL1Renderer){console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: can only be used with WebGL2.");return}const xt=S.max.x-S.min.x+1,yt=S.max.y-S.min.y+1,Et=S.max.z-S.min.z+1,At=F.convert(U.format),Pt=F.convert(U.type);let Rt;if(U.isData3DTexture)Tt.setTexture3D(U,0),Rt=32879;else if(U.isDataArrayTexture)Tt.setTexture2DArray(U,0),Rt=35866;else{console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: only supports THREE.DataTexture3D and THREE.DataTexture2DArray.");return}z.pixelStorei(37440,U.flipY),z.pixelStorei(37441,U.premultiplyAlpha),z.pixelStorei(3317,U.unpackAlignment);const Dt=z.getParameter(3314),zt=z.getParameter(32878),he=z.getParameter(3316),Te=z.getParameter(3315),nn=z.getParameter(32877),Jt=k.isCompressedTexture?k.mipmaps[0]:k.image;z.pixelStorei(3314,Jt.width),z.pixelStorei(32878,Jt.height),z.pixelStorei(3316,S.min.x),z.pixelStorei(3315,S.min.y),z.pixelStorei(32877,S.min.z),k.isDataTexture||k.isData3DTexture?z.texSubImage3D(Rt,W,O.x,O.y,O.z,xt,yt,Et,At,Pt,Jt.data):k.isCompressedArrayTexture?(console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: untested support for compressed srcTexture."),z.compressedTexSubImage3D(Rt,W,O.x,O.y,O.z,xt,yt,Et,At,Jt.data)):z.texSubImage3D(Rt,W,O.x,O.y,O.z,xt,yt,Et,At,Pt,Jt),z.pixelStorei(3314,Dt),z.pixelStorei(32878,zt),z.pixelStorei(3316,he),z.pixelStorei(3315,Te),z.pixelStorei(32877,nn),W===0&&U.generateMipmaps&&z.generateMipmap(Rt),Mt.unbindTexture()},this.initTexture=function(S){S.isCubeTexture?Tt.setTextureCube(S,0):S.isData3DTexture?Tt.setTexture3D(S,0):S.isDataArrayTexture||S.isCompressedArrayTexture?Tt.setTexture2DArray(S,0):Tt.setTexture2D(S,0),Mt.unbindTexture()},this.resetState=function(){x=0,b=0,y=null,Mt.reset(),ot.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get physicallyCorrectLights(){return console.warn("THREE.WebGLRenderer: the property .physicallyCorrectLights has been removed. Set renderer.useLegacyLights instead."),!this.useLegacyLights}set physicallyCorrectLights(t){console.warn("THREE.WebGLRenderer: the property .physicallyCorrectLights has been removed. Set renderer.useLegacyLights instead."),this.useLegacyLights=!t}}class Qd extends Ya{}Qd.prototype.isWebGL1Renderer=!0;class Br{constructor(t,e=1,n=1e3){this.isFog=!0,this.name="",this.color=new Ot(t),this.near=e,this.far=n}clone(){return new Br(this.color,this.near,this.far)}toJSON(){return{type:"Fog",color:this.color.getHex(),near:this.near,far:this.far}}}class tf extends re{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),t.background!==null&&(this.background=t.background.clone()),t.environment!==null&&(this.environment=t.environment.clone()),t.fog!==null&&(this.fog=t.fog.clone()),this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,t.overrideMaterial!==null&&(this.overrideMaterial=t.overrideMaterial.clone()),this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){const e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(e.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(e.object.backgroundIntensity=this.backgroundIntensity),e}get autoUpdate(){return console.warn("THREE.Scene: autoUpdate was renamed to matrixWorldAutoUpdate in r144."),this.matrixWorldAutoUpdate}set autoUpdate(t){console.warn("THREE.Scene: autoUpdate was renamed to matrixWorldAutoUpdate in r144."),this.matrixWorldAutoUpdate=t}}class Be{constructor(){this.type="Curve",this.arcLengthDivisions=200}getPoint(){return console.warn("THREE.Curve: .getPoint() not implemented."),null}getPointAt(t,e){const n=this.getUtoTmapping(t);return this.getPoint(n,e)}getPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return e}getSpacedPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPointAt(n/t));return e}getLength(){const t=this.getLengths();return t[t.length-1]}getLengths(t=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===t+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const e=[];let n,i=this.getPoint(0),r=0;e.push(0);for(let o=1;o<=t;o++)n=this.getPoint(o/t),r+=n.distanceTo(i),e.push(r),i=n;return this.cacheArcLengths=e,e}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(t,e){const n=this.getLengths();let i=0;const r=n.length;let o;e?o=e:o=t*n[r-1];let s=0,c=r-1,l;for(;s<=c;)if(i=Math.floor(s+(c-s)/2),l=n[i]-o,l<0)s=i+1;else if(l>0)c=i-1;else{c=i;break}if(i=c,n[i]===o)return i/(r-1);const h=n[i],f=n[i+1]-h,p=(o-h)/f;return(i+p)/(r-1)}getTangent(t,e){let i=t-1e-4,r=t+1e-4;i<0&&(i=0),r>1&&(r=1);const o=this.getPoint(i),s=this.getPoint(r),c=e||(o.isVector2?new dt:new P);return c.copy(s).sub(o).normalize(),c}getTangentAt(t,e){const n=this.getUtoTmapping(t);return this.getTangent(n,e)}computeFrenetFrames(t,e){const n=new P,i=[],r=[],o=[],s=new P,c=new Kt;for(let p=0;p<=t;p++){const g=p/t;i[p]=this.getTangentAt(g,new P)}r[0]=new P,o[0]=new P;let l=Number.MAX_VALUE;const h=Math.abs(i[0].x),d=Math.abs(i[0].y),f=Math.abs(i[0].z);h<=l&&(l=h,n.set(1,0,0)),d<=l&&(l=d,n.set(0,1,0)),f<=l&&n.set(0,0,1),s.crossVectors(i[0],n).normalize(),r[0].crossVectors(i[0],s),o[0].crossVectors(i[0],r[0]);for(let p=1;p<=t;p++){if(r[p]=r[p-1].clone(),o[p]=o[p-1].clone(),s.crossVectors(i[p-1],i[p]),s.length()>Number.EPSILON){s.normalize();const g=Math.acos(ne(i[p-1].dot(i[p]),-1,1));r[p].applyMatrix4(c.makeRotationAxis(s,g))}o[p].crossVectors(i[p],r[p])}if(e===!0){let p=Math.acos(ne(r[0].dot(r[t]),-1,1));p/=t,i[0].dot(s.crossVectors(r[0],r[t]))>0&&(p=-p);for(let g=1;g<=t;g++)r[g].applyMatrix4(c.makeRotationAxis(i[g],p*g)),o[g].crossVectors(i[g],r[g])}return{tangents:i,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}toJSON(){const t={metadata:{version:4.5,type:"Curve",generator:"Curve.toJSON"}};return t.arcLengthDivisions=this.arcLengthDivisions,t.type=this.type,t}fromJSON(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}}class Gr extends Be{constructor(t=0,e=0,n=1,i=1,r=0,o=Math.PI*2,s=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=t,this.aY=e,this.xRadius=n,this.yRadius=i,this.aStartAngle=r,this.aEndAngle=o,this.aClockwise=s,this.aRotation=c}getPoint(t,e){const n=e||new dt,i=Math.PI*2;let r=this.aEndAngle-this.aStartAngle;const o=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=i;for(;r>i;)r-=i;r<Number.EPSILON&&(o?r=0:r=i),this.aClockwise===!0&&!o&&(r===i?r=-i:r=r-i);const s=this.aStartAngle+t*r;let c=this.aX+this.xRadius*Math.cos(s),l=this.aY+this.yRadius*Math.sin(s);if(this.aRotation!==0){const h=Math.cos(this.aRotation),d=Math.sin(this.aRotation),f=c-this.aX,p=l-this.aY;c=f*h-p*d+this.aX,l=f*d+p*h+this.aY}return n.set(c,l)}copy(t){return super.copy(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}toJSON(){const t=super.toJSON();return t.aX=this.aX,t.aY=this.aY,t.xRadius=this.xRadius,t.yRadius=this.yRadius,t.aStartAngle=this.aStartAngle,t.aEndAngle=this.aEndAngle,t.aClockwise=this.aClockwise,t.aRotation=this.aRotation,t}fromJSON(t){return super.fromJSON(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}}class ef extends Gr{constructor(t,e,n,i,r,o){super(t,e,n,n,i,r,o),this.isArcCurve=!0,this.type="ArcCurve"}}function Vr(){let a=0,t=0,e=0,n=0;function i(r,o,s,c){a=r,t=s,e=-3*r+3*o-2*s-c,n=2*r-2*o+s+c}return{initCatmullRom:function(r,o,s,c,l){i(o,s,l*(s-r),l*(c-o))},initNonuniformCatmullRom:function(r,o,s,c,l,h,d){let f=(o-r)/l-(s-r)/(l+h)+(s-o)/h,p=(s-o)/h-(c-o)/(h+d)+(c-s)/d;f*=h,p*=h,i(o,s,f,p)},calc:function(r){const o=r*r,s=o*r;return a+t*r+e*o+n*s}}}const Ii=new P,Mr=new Vr,Sr=new Vr,yr=new Vr;class nf extends Be{constructor(t=[],e=!1,n="centripetal",i=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=t,this.closed=e,this.curveType=n,this.tension=i}getPoint(t,e=new P){const n=e,i=this.points,r=i.length,o=(r-(this.closed?0:1))*t;let s=Math.floor(o),c=o-s;this.closed?s+=s>0?0:(Math.floor(Math.abs(s)/r)+1)*r:c===0&&s===r-1&&(s=r-2,c=1);let l,h;this.closed||s>0?l=i[(s-1)%r]:(Ii.subVectors(i[0],i[1]).add(i[0]),l=Ii);const d=i[s%r],f=i[(s+1)%r];if(this.closed||s+2<r?h=i[(s+2)%r]:(Ii.subVectors(i[r-1],i[r-2]).add(i[r-1]),h=Ii),this.curveType==="centripetal"||this.curveType==="chordal"){const p=this.curveType==="chordal"?.5:.25;let g=Math.pow(l.distanceToSquared(d),p),_=Math.pow(d.distanceToSquared(f),p),m=Math.pow(f.distanceToSquared(h),p);_<1e-4&&(_=1),g<1e-4&&(g=_),m<1e-4&&(m=_),Mr.initNonuniformCatmullRom(l.x,d.x,f.x,h.x,g,_,m),Sr.initNonuniformCatmullRom(l.y,d.y,f.y,h.y,g,_,m),yr.initNonuniformCatmullRom(l.z,d.z,f.z,h.z,g,_,m)}else this.curveType==="catmullrom"&&(Mr.initCatmullRom(l.x,d.x,f.x,h.x,this.tension),Sr.initCatmullRom(l.y,d.y,f.y,h.y,this.tension),yr.initCatmullRom(l.z,d.z,f.z,h.z,this.tension));return n.set(Mr.calc(c),Sr.calc(c),yr.calc(c)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const i=t.points[e];this.points.push(i.clone())}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const i=this.points[e];t.points.push(i.toArray())}return t.closed=this.closed,t.curveType=this.curveType,t.tension=this.tension,t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const i=t.points[e];this.points.push(new P().fromArray(i))}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}}function ua(a,t,e,n,i){const r=(n-t)*.5,o=(i-e)*.5,s=a*a,c=a*s;return(2*e-2*n+r+o)*c+(-3*e+3*n-2*r-o)*s+r*a+e}function rf(a,t){const e=1-a;return e*e*t}function sf(a,t){return 2*(1-a)*a*t}function af(a,t){return a*a*t}function ii(a,t,e,n){return rf(a,t)+sf(a,e)+af(a,n)}function of(a,t){const e=1-a;return e*e*e*t}function lf(a,t){const e=1-a;return 3*e*e*a*t}function cf(a,t){return 3*(1-a)*a*a*t}function hf(a,t){return a*a*a*t}function ri(a,t,e,n,i){return of(a,t)+lf(a,e)+cf(a,n)+hf(a,i)}class ja extends Be{constructor(t=new dt,e=new dt,n=new dt,i=new dt){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=t,this.v1=e,this.v2=n,this.v3=i}getPoint(t,e=new dt){const n=e,i=this.v0,r=this.v1,o=this.v2,s=this.v3;return n.set(ri(t,i.x,r.x,o.x,s.x),ri(t,i.y,r.y,o.y,s.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}class uf extends Be{constructor(t=new P,e=new P,n=new P,i=new P){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=t,this.v1=e,this.v2=n,this.v3=i}getPoint(t,e=new P){const n=e,i=this.v0,r=this.v1,o=this.v2,s=this.v3;return n.set(ri(t,i.x,r.x,o.x,s.x),ri(t,i.y,r.y,o.y,s.y),ri(t,i.z,r.z,o.z,s.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}class kr extends Be{constructor(t=new dt,e=new dt){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=t,this.v2=e}getPoint(t,e=new dt){const n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new dt){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class df extends Be{constructor(t=new P,e=new P){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=t,this.v2=e}getPoint(t,e=new P){const n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new P){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class Ja extends Be{constructor(t=new dt,e=new dt,n=new dt){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new dt){const n=e,i=this.v0,r=this.v1,o=this.v2;return n.set(ii(t,i.x,r.x,o.x),ii(t,i.y,r.y,o.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class ff extends Be{constructor(t=new P,e=new P,n=new P){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new P){const n=e,i=this.v0,r=this.v1,o=this.v2;return n.set(ii(t,i.x,r.x,o.x),ii(t,i.y,r.y,o.y),ii(t,i.z,r.z,o.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class Za extends Be{constructor(t=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=t}getPoint(t,e=new dt){const n=e,i=this.points,r=(i.length-1)*t,o=Math.floor(r),s=r-o,c=i[o===0?o:o-1],l=i[o],h=i[o>i.length-2?i.length-1:o+1],d=i[o>i.length-3?i.length-1:o+2];return n.set(ua(s,c.x,l.x,h.x,d.x),ua(s,c.y,l.y,h.y,d.y)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const i=t.points[e];this.points.push(i.clone())}return this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const i=this.points[e];t.points.push(i.toArray())}return t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const i=t.points[e];this.points.push(new dt().fromArray(i))}return this}}var pf=Object.freeze({__proto__:null,ArcCurve:ef,CatmullRomCurve3:nf,CubicBezierCurve:ja,CubicBezierCurve3:uf,EllipseCurve:Gr,LineCurve:kr,LineCurve3:df,QuadraticBezierCurve:Ja,QuadraticBezierCurve3:ff,SplineCurve:Za});class mf extends Be{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(t){this.curves.push(t)}closePath(){const t=this.curves[0].getPoint(0),e=this.curves[this.curves.length-1].getPoint(1);t.equals(e)||this.curves.push(new kr(e,t))}getPoint(t,e){const n=t*this.getLength(),i=this.getCurveLengths();let r=0;for(;r<i.length;){if(i[r]>=n){const o=i[r]-n,s=this.curves[r],c=s.getLength(),l=c===0?0:1-o/c;return s.getPointAt(l,e)}r++}return null}getLength(){const t=this.getCurveLengths();return t[t.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;const t=[];let e=0;for(let n=0,i=this.curves.length;n<i;n++)e+=this.curves[n].getLength(),t.push(e);return this.cacheLengths=t,t}getSpacedPoints(t=40){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return this.autoClose&&e.push(e[0]),e}getPoints(t=12){const e=[];let n;for(let i=0,r=this.curves;i<r.length;i++){const o=r[i],s=o.isEllipseCurve?t*2:o.isLineCurve||o.isLineCurve3?1:o.isSplineCurve?t*o.points.length:t,c=o.getPoints(s);for(let l=0;l<c.length;l++){const h=c[l];n&&n.equals(h)||(e.push(h),n=h)}}return this.autoClose&&e.length>1&&!e[e.length-1].equals(e[0])&&e.push(e[0]),e}copy(t){super.copy(t),this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const i=t.curves[e];this.curves.push(i.clone())}return this.autoClose=t.autoClose,this}toJSON(){const t=super.toJSON();t.autoClose=this.autoClose,t.curves=[];for(let e=0,n=this.curves.length;e<n;e++){const i=this.curves[e];t.curves.push(i.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.autoClose=t.autoClose,this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const i=t.curves[e];this.curves.push(new pf[i.type]().fromJSON(i))}return this}}class gf extends mf{constructor(t){super(),this.type="Path",this.currentPoint=new dt,t&&this.setFromPoints(t)}setFromPoints(t){this.moveTo(t[0].x,t[0].y);for(let e=1,n=t.length;e<n;e++)this.lineTo(t[e].x,t[e].y);return this}moveTo(t,e){return this.currentPoint.set(t,e),this}lineTo(t,e){const n=new kr(this.currentPoint.clone(),new dt(t,e));return this.curves.push(n),this.currentPoint.set(t,e),this}quadraticCurveTo(t,e,n,i){const r=new Ja(this.currentPoint.clone(),new dt(t,e),new dt(n,i));return this.curves.push(r),this.currentPoint.set(n,i),this}bezierCurveTo(t,e,n,i,r,o){const s=new ja(this.currentPoint.clone(),new dt(t,e),new dt(n,i),new dt(r,o));return this.curves.push(s),this.currentPoint.set(r,o),this}splineThru(t){const e=[this.currentPoint.clone()].concat(t),n=new Za(e);return this.curves.push(n),this.currentPoint.copy(t[t.length-1]),this}arc(t,e,n,i,r,o){const s=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(t+s,e+c,n,i,r,o),this}absarc(t,e,n,i,r,o){return this.absellipse(t,e,n,n,i,r,o),this}ellipse(t,e,n,i,r,o,s,c){const l=this.currentPoint.x,h=this.currentPoint.y;return this.absellipse(t+l,e+h,n,i,r,o,s,c),this}absellipse(t,e,n,i,r,o,s,c){const l=new Gr(t,e,n,i,r,o,s,c);if(this.curves.length>0){const d=l.getPoint(0);d.equals(this.currentPoint)||this.lineTo(d.x,d.y)}this.curves.push(l);const h=l.getPoint(1);return this.currentPoint.copy(h),this}copy(t){return super.copy(t),this.currentPoint.copy(t.currentPoint),this}toJSON(){const t=super.toJSON();return t.currentPoint=this.currentPoint.toArray(),t}fromJSON(t){return super.fromJSON(t),this.currentPoint.fromArray(t.currentPoint),this}}class Hr extends Ee{constructor(t=[new dt(0,-.5),new dt(.5,0),new dt(0,.5)],e=12,n=0,i=Math.PI*2){super(),this.type="LatheGeometry",this.parameters={points:t,segments:e,phiStart:n,phiLength:i},e=Math.floor(e),i=ne(i,0,Math.PI*2);const r=[],o=[],s=[],c=[],l=[],h=1/e,d=new P,f=new dt,p=new P,g=new P,_=new P;let m=0,u=0;for(let E=0;E<=t.length-1;E++)switch(E){case 0:m=t[E+1].x-t[E].x,u=t[E+1].y-t[E].y,p.x=u*1,p.y=-m,p.z=u*0,_.copy(p),p.normalize(),c.push(p.x,p.y,p.z);break;case t.length-1:c.push(_.x,_.y,_.z);break;default:m=t[E+1].x-t[E].x,u=t[E+1].y-t[E].y,p.x=u*1,p.y=-m,p.z=u*0,g.copy(p),p.x+=_.x,p.y+=_.y,p.z+=_.z,p.normalize(),c.push(p.x,p.y,p.z),_.copy(g)}for(let E=0;E<=e;E++){const x=n+E*h*i,b=Math.sin(x),y=Math.cos(x);for(let C=0;C<=t.length-1;C++){d.x=t[C].x*b,d.y=t[C].y,d.z=t[C].x*y,o.push(d.x,d.y,d.z),f.x=E/e,f.y=C/(t.length-1),s.push(f.x,f.y);const D=c[3*C+0]*b,R=c[3*C+1],M=c[3*C+0]*y;l.push(D,R,M)}}for(let E=0;E<e;E++)for(let x=0;x<t.length-1;x++){const b=x+E*t.length,y=b,C=b+t.length,D=b+t.length+1,R=b+1;r.push(y,C,R),r.push(D,R,C)}this.setIndex(r),this.setAttribute("position",new se(o,3)),this.setAttribute("uv",new se(s,2)),this.setAttribute("normal",new se(l,3))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Hr(t.points,t.segments,t.phiStart,t.phiLength)}}class Wr extends Hr{constructor(t=1,e=1,n=4,i=8){const r=new gf;r.absarc(0,-e/2,t,Math.PI*1.5,0),r.absarc(0,e/2,t,0,Math.PI*.5),super(r.getPoints(n),i),this.type="CapsuleGeometry",this.parameters={radius:t,height:e,capSegments:n,radialSegments:i}}static fromJSON(t){return new Wr(t.radius,t.length,t.capSegments,t.radialSegments)}}class qr extends Ee{constructor(t=1,e=1,n=1,i=32,r=1,o=!1,s=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:t,radiusBottom:e,height:n,radialSegments:i,heightSegments:r,openEnded:o,thetaStart:s,thetaLength:c};const l=this;i=Math.floor(i),r=Math.floor(r);const h=[],d=[],f=[],p=[];let g=0;const _=[],m=n/2;let u=0;E(),o===!1&&(t>0&&x(!0),e>0&&x(!1)),this.setIndex(h),this.setAttribute("position",new se(d,3)),this.setAttribute("normal",new se(f,3)),this.setAttribute("uv",new se(p,2));function E(){const b=new P,y=new P;let C=0;const D=(e-t)/n;for(let R=0;R<=r;R++){const M=[],T=R/r,X=T*(e-t)+t;for(let q=0;q<=i;q++){const I=q/i,N=I*c+s,B=Math.sin(N),Q=Math.cos(N);y.x=X*B,y.y=-T*n+m,y.z=X*Q,d.push(y.x,y.y,y.z),b.set(B,D,Q).normalize(),f.push(b.x,b.y,b.z),p.push(I,1-T),M.push(g++)}_.push(M)}for(let R=0;R<i;R++)for(let M=0;M<r;M++){const T=_[M][R],X=_[M+1][R],q=_[M+1][R+1],I=_[M][R+1];h.push(T,X,I),h.push(X,q,I),C+=6}l.addGroup(u,C,0),u+=C}function x(b){const y=g,C=new dt,D=new P;let R=0;const M=b===!0?t:e,T=b===!0?1:-1;for(let q=1;q<=i;q++)d.push(0,m*T,0),f.push(0,T,0),p.push(.5,.5),g++;const X=g;for(let q=0;q<=i;q++){const N=q/i*c+s,B=Math.cos(N),Q=Math.sin(N);D.x=M*Q,D.y=m*T,D.z=M*B,d.push(D.x,D.y,D.z),f.push(0,T,0),C.x=B*.5+.5,C.y=Q*.5*T+.5,p.push(C.x,C.y),g++}for(let q=0;q<i;q++){const I=y+q,N=X+q;b===!0?h.push(N,N+1,I):h.push(N+1,N,I),R+=3}l.addGroup(u,R,b===!0?1:2),u+=R}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new qr(t.radiusTop,t.radiusBottom,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}}class Xr extends Ee{constructor(t=1,e=32,n=16,i=0,r=Math.PI*2,o=0,s=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:n,phiStart:i,phiLength:r,thetaStart:o,thetaLength:s},e=Math.max(3,Math.floor(e)),n=Math.max(2,Math.floor(n));const c=Math.min(o+s,Math.PI);let l=0;const h=[],d=new P,f=new P,p=[],g=[],_=[],m=[];for(let u=0;u<=n;u++){const E=[],x=u/n;let b=0;u===0&&o===0?b=.5/e:u===n&&c===Math.PI&&(b=-.5/e);for(let y=0;y<=e;y++){const C=y/e;d.x=-t*Math.cos(i+C*r)*Math.sin(o+x*s),d.y=t*Math.cos(o+x*s),d.z=t*Math.sin(i+C*r)*Math.sin(o+x*s),g.push(d.x,d.y,d.z),f.copy(d).normalize(),_.push(f.x,f.y,f.z),m.push(C+b,1-x),E.push(l++)}h.push(E)}for(let u=0;u<n;u++)for(let E=0;E<e;E++){const x=h[u][E+1],b=h[u][E],y=h[u+1][E],C=h[u+1][E+1];(u!==0||o>0)&&p.push(x,b,C),(u!==n-1||c<Math.PI)&&p.push(b,y,C)}this.setIndex(p),this.setAttribute("position",new se(g,3)),this.setAttribute("normal",new se(_,3)),this.setAttribute("uv",new se(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Xr(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}}class da extends hi{constructor(t){super(),this.isMeshStandardMaterial=!0,this.defines={STANDARD:""},this.type="MeshStandardMaterial",this.color=new Ot(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Ot(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Aa,this.normalScale=new dt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapIntensity=t.envMapIntensity,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}}class Ka extends re{constructor(t,e=1){super(),this.isLight=!0,this.type="Light",this.color=new Ot(t),this.intensity=e}dispose(){}copy(t,e){return super.copy(t,e),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){const e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,this.groundColor!==void 0&&(e.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(e.object.distance=this.distance),this.angle!==void 0&&(e.object.angle=this.angle),this.decay!==void 0&&(e.object.decay=this.decay),this.penumbra!==void 0&&(e.object.penumbra=this.penumbra),this.shadow!==void 0&&(e.object.shadow=this.shadow.toJSON()),e}}class _f extends Ka{constructor(t,e,n){super(t,n),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(re.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Ot(e)}copy(t,e){return super.copy(t,e),this.groundColor.copy(t.groundColor),this}}const br=new Kt,fa=new P,pa=new P;class vf{constructor(t){this.camera=t,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new dt(512,512),this.map=null,this.mapPass=null,this.matrix=new Kt,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Fr,this._frameExtents=new dt(1,1),this._viewportCount=1,this._viewports=[new ie(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(t){const e=this.camera,n=this.matrix;fa.setFromMatrixPosition(t.matrixWorld),e.position.copy(fa),pa.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(pa),e.updateMatrixWorld(),br.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),this._frustum.setFromProjectionMatrix(br),n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(br)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.bias=t.bias,this.radius=t.radius,this.mapSize.copy(t.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const t={};return this.bias!==0&&(t.bias=this.bias),this.normalBias!==0&&(t.normalBias=this.normalBias),this.radius!==1&&(t.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(t.mapSize=this.mapSize.toArray()),t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}}class xf extends vf{constructor(){super(new ka(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}}class Mf extends Ka{constructor(t,e){super(t,e),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(re.DEFAULT_UP),this.updateMatrix(),this.target=new re,this.shadow=new xf}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}}class Sf{constructor(t,e,n=0,i=1/0){this.ray=new Ia(t,e),this.near=n,this.far=i,this.camera=null,this.layers=new Or,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(t,e){this.ray.set(t,e)}setFromCamera(t,e){e.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(t.x,t.y,.5).unproject(e).sub(this.ray.origin).normalize(),this.camera=e):e.isOrthographicCamera?(this.ray.origin.set(t.x,t.y,(e.near+e.far)/(e.near-e.far)).unproject(e),this.ray.direction.set(0,0,-1).transformDirection(e.matrixWorld),this.camera=e):console.error("THREE.Raycaster: Unsupported camera type: "+e.type)}intersectObject(t,e=!0,n=[]){return Ur(t,this,n,e),n.sort(ma),n}intersectObjects(t,e=!0,n=[]){for(let i=0,r=t.length;i<r;i++)Ur(t[i],this,n,e);return n.sort(ma),n}}function ma(a,t){return a.distance-t.distance}function Ur(a,t,e,n){if(a.layers.test(t.layers)&&a.raycast(t,e),n===!0){const i=a.children;for(let r=0,o=i.length;r<o;r++)Ur(i[r],t,e,!0)}}class ga{constructor(t=1,e=0,n=0){return this.radius=t,this.phi=e,this.theta=n,this}set(t,e,n){return this.radius=t,this.phi=e,this.theta=n,this}copy(t){return this.radius=t.radius,this.phi=t.phi,this.theta=t.theta,this}makeSafe(){return this.phi=Math.max(1e-6,Math.min(Math.PI-1e-6,this.phi)),this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+e*e+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(t,n),this.phi=Math.acos(ne(e/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"151"}}));typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="151");const _a={type:"change"},wr={type:"start"},va={type:"end"};class yf extends Mn{constructor(t,e){super(),this.object=t,this.domElement=e,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new P,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:Sn.ROTATE,MIDDLE:Sn.DOLLY,RIGHT:Sn.PAN},this.touches={ONE:yn.ROTATE,TWO:yn.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return s.phi},this.getAzimuthalAngle=function(){return s.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(A){A.addEventListener("keydown",Bt),this._domElementKeyEvents=A},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",Bt),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,n.object.updateProjectionMatrix(),n.dispatchEvent(_a),n.update(),r=i.NONE},this.update=(function(){const A=new P,Y=new vn().setFromUnitVectors(t.up,new P(0,1,0)),F=Y.clone().invert(),ot=new P,ft=new vn,mt=2*Math.PI;return function(){const pt=n.object.position;A.copy(pt).sub(n.target),A.applyQuaternion(Y),s.setFromVector3(A),n.autoRotate&&r===i.NONE&&M(D()),n.enableDamping?(s.theta+=c.theta*n.dampingFactor,s.phi+=c.phi*n.dampingFactor):(s.theta+=c.theta,s.phi+=c.phi);let wt=n.minAzimuthAngle,Lt=n.maxAzimuthAngle;return isFinite(wt)&&isFinite(Lt)&&(wt<-Math.PI?wt+=mt:wt>Math.PI&&(wt-=mt),Lt<-Math.PI?Lt+=mt:Lt>Math.PI&&(Lt-=mt),wt<=Lt?s.theta=Math.max(wt,Math.min(Lt,s.theta)):s.theta=s.theta>(wt+Lt)/2?Math.max(wt,s.theta):Math.min(Lt,s.theta)),s.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,s.phi)),s.makeSafe(),s.radius*=l,s.radius=Math.max(n.minDistance,Math.min(n.maxDistance,s.radius)),n.enableDamping===!0?n.target.addScaledVector(h,n.dampingFactor):n.target.add(h),A.setFromSpherical(s),A.applyQuaternion(F),pt.copy(n.target).add(A),n.object.lookAt(n.target),n.enableDamping===!0?(c.theta*=1-n.dampingFactor,c.phi*=1-n.dampingFactor,h.multiplyScalar(1-n.dampingFactor)):(c.set(0,0,0),h.set(0,0,0)),l=1,d||ot.distanceToSquared(n.object.position)>o||8*(1-ft.dot(n.object.quaternion))>o?(n.dispatchEvent(_a),ot.copy(n.object.position),ft.copy(n.object.quaternion),d=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",v),n.domElement.removeEventListener("pointerdown",Tt),n.domElement.removeEventListener("pointercancel",kt),n.domElement.removeEventListener("wheel",Nt),n.domElement.removeEventListener("pointermove",Wt),n.domElement.removeEventListener("pointerup",kt),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",Bt),n._domElementKeyEvents=null)};const n=this,i={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=i.NONE;const o=1e-6,s=new ga,c=new ga;let l=1;const h=new P;let d=!1;const f=new dt,p=new dt,g=new dt,_=new dt,m=new dt,u=new dt,E=new dt,x=new dt,b=new dt,y=[],C={};function D(){return 2*Math.PI/60/60*n.autoRotateSpeed}function R(){return Math.pow(.95,n.zoomSpeed)}function M(A){c.theta-=A}function T(A){c.phi-=A}const X=(function(){const A=new P;return function(F,ot){A.setFromMatrixColumn(ot,0),A.multiplyScalar(-F),h.add(A)}})(),q=(function(){const A=new P;return function(F,ot){n.screenSpacePanning===!0?A.setFromMatrixColumn(ot,1):(A.setFromMatrixColumn(ot,0),A.crossVectors(n.object.up,A)),A.multiplyScalar(F),h.add(A)}})(),I=(function(){const A=new P;return function(F,ot){const ft=n.domElement;if(n.object.isPerspectiveCamera){const mt=n.object.position;A.copy(mt).sub(n.target);let ht=A.length();ht*=Math.tan(n.object.fov/2*Math.PI/180),X(2*F*ht/ft.clientHeight,n.object.matrix),q(2*ot*ht/ft.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(X(F*(n.object.right-n.object.left)/n.object.zoom/ft.clientWidth,n.object.matrix),q(ot*(n.object.top-n.object.bottom)/n.object.zoom/ft.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function N(A){n.object.isPerspectiveCamera?l/=A:n.object.isOrthographicCamera?(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom*A)),n.object.updateProjectionMatrix(),d=!0):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function B(A){n.object.isPerspectiveCamera?l*=A:n.object.isOrthographicCamera?(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/A)),n.object.updateProjectionMatrix(),d=!0):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function Q(A){f.set(A.clientX,A.clientY)}function et(A){E.set(A.clientX,A.clientY)}function j(A){_.set(A.clientX,A.clientY)}function Z(A){p.set(A.clientX,A.clientY),g.subVectors(p,f).multiplyScalar(n.rotateSpeed);const Y=n.domElement;M(2*Math.PI*g.x/Y.clientHeight),T(2*Math.PI*g.y/Y.clientHeight),f.copy(p),n.update()}function nt(A){x.set(A.clientX,A.clientY),b.subVectors(x,E),b.y>0?N(R()):b.y<0&&B(R()),E.copy(x),n.update()}function vt(A){m.set(A.clientX,A.clientY),u.subVectors(m,_).multiplyScalar(n.panSpeed),I(u.x,u.y),_.copy(m),n.update()}function ct(A){A.deltaY<0?B(R()):A.deltaY>0&&N(R()),n.update()}function V(A){let Y=!1;switch(A.code){case n.keys.UP:A.ctrlKey||A.metaKey||A.shiftKey?T(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):I(0,n.keyPanSpeed),Y=!0;break;case n.keys.BOTTOM:A.ctrlKey||A.metaKey||A.shiftKey?T(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):I(0,-n.keyPanSpeed),Y=!0;break;case n.keys.LEFT:A.ctrlKey||A.metaKey||A.shiftKey?M(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):I(n.keyPanSpeed,0),Y=!0;break;case n.keys.RIGHT:A.ctrlKey||A.metaKey||A.shiftKey?M(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):I(-n.keyPanSpeed,0),Y=!0;break}Y&&(A.preventDefault(),n.update())}function J(){if(y.length===1)f.set(y[0].pageX,y[0].pageY);else{const A=.5*(y[0].pageX+y[1].pageX),Y=.5*(y[0].pageY+y[1].pageY);f.set(A,Y)}}function st(){if(y.length===1)_.set(y[0].pageX,y[0].pageY);else{const A=.5*(y[0].pageX+y[1].pageX),Y=.5*(y[0].pageY+y[1].pageY);_.set(A,Y)}}function lt(){const A=y[0].pageX-y[1].pageX,Y=y[0].pageY-y[1].pageY,F=Math.sqrt(A*A+Y*Y);E.set(0,F)}function z(){n.enableZoom&&lt(),n.enablePan&&st()}function bt(){n.enableZoom&&lt(),n.enableRotate&&J()}function St(A){if(y.length==1)p.set(A.pageX,A.pageY);else{const F=at(A),ot=.5*(A.pageX+F.x),ft=.5*(A.pageY+F.y);p.set(ot,ft)}g.subVectors(p,f).multiplyScalar(n.rotateSpeed);const Y=n.domElement;M(2*Math.PI*g.x/Y.clientHeight),T(2*Math.PI*g.y/Y.clientHeight),f.copy(p)}function it(A){if(y.length===1)m.set(A.pageX,A.pageY);else{const Y=at(A),F=.5*(A.pageX+Y.x),ot=.5*(A.pageY+Y.y);m.set(F,ot)}u.subVectors(m,_).multiplyScalar(n.panSpeed),I(u.x,u.y),_.copy(m)}function Mt(A){const Y=at(A),F=A.pageX-Y.x,ot=A.pageY-Y.y,ft=Math.sqrt(F*F+ot*ot);x.set(0,ft),b.set(0,Math.pow(x.y/E.y,n.zoomSpeed)),N(b.y),E.copy(x)}function Ft(A){n.enableZoom&&Mt(A),n.enablePan&&it(A)}function gt(A){n.enableZoom&&Mt(A),n.enableRotate&&St(A)}function Tt(A){n.enabled!==!1&&(y.length===0&&(n.domElement.setPointerCapture(A.pointerId),n.domElement.addEventListener("pointermove",Wt),n.domElement.addEventListener("pointerup",kt)),G(A),A.pointerType==="touch"?ae(A):jt(A))}function Wt(A){n.enabled!==!1&&(A.pointerType==="touch"?w(A):qt(A))}function kt(A){K(A),y.length===0&&(n.domElement.releasePointerCapture(A.pointerId),n.domElement.removeEventListener("pointermove",Wt),n.domElement.removeEventListener("pointerup",kt)),n.dispatchEvent(va),r=i.NONE}function jt(A){let Y;switch(A.button){case 0:Y=n.mouseButtons.LEFT;break;case 1:Y=n.mouseButtons.MIDDLE;break;case 2:Y=n.mouseButtons.RIGHT;break;default:Y=-1}switch(Y){case Sn.DOLLY:if(n.enableZoom===!1)return;et(A),r=i.DOLLY;break;case Sn.ROTATE:if(A.ctrlKey||A.metaKey||A.shiftKey){if(n.enablePan===!1)return;j(A),r=i.PAN}else{if(n.enableRotate===!1)return;Q(A),r=i.ROTATE}break;case Sn.PAN:if(A.ctrlKey||A.metaKey||A.shiftKey){if(n.enableRotate===!1)return;Q(A),r=i.ROTATE}else{if(n.enablePan===!1)return;j(A),r=i.PAN}break;default:r=i.NONE}r!==i.NONE&&n.dispatchEvent(wr)}function qt(A){switch(r){case i.ROTATE:if(n.enableRotate===!1)return;Z(A);break;case i.DOLLY:if(n.enableZoom===!1)return;nt(A);break;case i.PAN:if(n.enablePan===!1)return;vt(A);break}}function Nt(A){n.enabled===!1||n.enableZoom===!1||r!==i.NONE||(A.preventDefault(),n.dispatchEvent(wr),ct(A),n.dispatchEvent(va))}function Bt(A){n.enabled===!1||n.enablePan===!1||V(A)}function ae(A){switch(tt(A),y.length){case 1:switch(n.touches.ONE){case yn.ROTATE:if(n.enableRotate===!1)return;J(),r=i.TOUCH_ROTATE;break;case yn.PAN:if(n.enablePan===!1)return;st(),r=i.TOUCH_PAN;break;default:r=i.NONE}break;case 2:switch(n.touches.TWO){case yn.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;z(),r=i.TOUCH_DOLLY_PAN;break;case yn.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;bt(),r=i.TOUCH_DOLLY_ROTATE;break;default:r=i.NONE}break;default:r=i.NONE}r!==i.NONE&&n.dispatchEvent(wr)}function w(A){switch(tt(A),r){case i.TOUCH_ROTATE:if(n.enableRotate===!1)return;St(A),n.update();break;case i.TOUCH_PAN:if(n.enablePan===!1)return;it(A),n.update();break;case i.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Ft(A),n.update();break;case i.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;gt(A),n.update();break;default:r=i.NONE}}function v(A){n.enabled!==!1&&A.preventDefault()}function G(A){y.push(A)}function K(A){delete C[A.pointerId];for(let Y=0;Y<y.length;Y++)if(y[Y].pointerId==A.pointerId){y.splice(Y,1);return}}function tt(A){let Y=C[A.pointerId];Y===void 0&&(Y=new dt,C[A.pointerId]=Y),Y.set(A.pageX,A.pageY)}function at(A){const Y=A.pointerId===y[0].pointerId?y[1]:y[0];return C[Y.pointerId]}n.domElement.addEventListener("contextmenu",v),n.domElement.addEventListener("pointerdown",Tt),n.domElement.addEventListener("pointercancel",kt),n.domElement.addEventListener("wheel",Nt,{passive:!1}),this.update()}}function Er(a,t,e,n){a.position.fromArray(t,n*3),a.quaternion.set(e[n*4+1],e[n*4+2],e[n*4+3],e[n*4])}function bf(a,t,e=new Map){const n=a.geom_type[t],i=Array.from(a.geom_size.subarray(t*3,t*3+3)),r=a.geom_dataid[t],o=`${n}:${n===1||n===7?r:i.join(",")}`;if(e.has(o))return e.get(o);let s;if(n===7){s=new Ee;const c=a.mesh_vertadr[r]*3,l=c+a.mesh_vertnum[r]*3;s.setAttribute("position",new Ie(Float32Array.from(a.mesh_vert.subarray(c,l)),3));const h=a.mesh_faceadr[r]*3;s.setIndex(Array.from(a.mesh_face.subarray(h,h+a.mesh_facenum[r]*3))),s.computeVertexNormals()}else if(n===1){const c=a.hfield_nrow[r],l=a.hfield_ncol[r],[h,d,f,p]=a.hfield_size.subarray(r*4,r*4+4),g=a.hfield_adr[r],_=[],m=[];for(let x=0;x<c;x++)for(let b=0;b<l;b++)if(_.push((b/(l-1)*2-1)*h,(x/(c-1)*2-1)*d,a.hfield_data[g+x*l+b]*f),x<c-1&&b<l-1){const y=x*l+b,C=y+1,D=y+l,R=D+1;m.push(y,C,R,y,R,D)}const u=[];for(let x=0;x<l;x++)u.push(x);for(let x=1;x<c;x++)u.push(x*l+l-1);for(let x=l-2;x>=0;x--)u.push((c-1)*l+x);for(let x=c-2;x>0;x--)u.push(x*l);const E=_.length/3;u.forEach(x=>_.push(_[x*3],_[x*3+1],-p)),u.forEach((x,b)=>{const y=(b+1)%u.length;m.push(x,E+b,E+y,x,E+y,u[y])}),s=new Ee,s.setAttribute("position",new se(_,3)),s.setIndex(m),s.computeVertexNormals()}else if(n===0)s=new zi(i[0]?i[0]*2:200,i[1]?i[1]*2:200);else if(n===2||n===4)s=new Xr(n===2?i[0]:1,24,16),n===4&&s.scale(...i);else if(n===3)s=new Wr(i[0],2*i[1],8,16),s.rotateX(Math.PI/2);else if(n===5)s=new qr(i[0],i[0],2*i[1],24),s.rotateX(Math.PI/2);else if(n===6)s=new jn(i[0]*2,i[1]*2,i[2]*2);else throw new Error(`Unsupported MuJoCo visual geometry ${n}`);return s.computeBoundingSphere(),e.set(o,s),s}class wf{constructor({physics:t,container:e}){this.physics=t,this.container=e,this.scene=new tf,this.scene.background=new Ot("#b7c5d0"),this.scene.fog=new Br("#b7c5d0",100,220),this.camera=new we(48,1,.025,300),this.camera.up.set(0,0,1),this.renderer=new Ya({antialias:!0,alpha:!1}),this.renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.75)),this.renderer.outputEncoding=Gt,this.renderer.toneMapping=wa,this.renderer.toneMappingExposure=.95,this.renderer.shadowMap.enabled=!0,this.renderer.shadowMap.type=Ma,e.appendChild(this.renderer.domElement),this.renderer.domElement.style.display="block",this.controls=new yf(this.camera,this.renderer.domElement),this.controls.enableDamping=!0,this.controls.dampingFactor=.09,this.controls.minDistance=1,this.controls.maxDistance=135,this.controls.maxPolarAngle=Math.PI*.49,this.scene.add(new _f("#ecf4ff","#667276",.7)),this.sun=new Mf("#fff5df",1.4),this.sun.castShadow=!0,this.sun.shadow.mapSize.set(2048,2048),Object.assign(this.sun.shadow.camera,{left:-8,right:8,top:8,bottom:-8,near:.1,far:50}),this.sun.shadow.normalBias=.015,this.sun.shadow.bias=-1e-4,this.scene.add(this.sun,this.sun.target),this.geometryCache=new Map,this.materials=[],this.bodies=new Map,this.robotRoot=new Ye,this.terrainRoot=new Ye,this.ghostRoot=new Ye,this.ghostBodies=new Map,this.ghostRoot.visible=!1,this.scene.add(this.terrainRoot,this.robotRoot,this.ghostRoot),this.ghostMaterial=new da({color:"#51e6ef",transparent:!0,opacity:.32,roughness:.55,depthWrite:!1}),this.materials.push(this.ghostMaterial);const n=t.model;for(let i=0;i<n.ngeom;i++){const r=n.geom_group[i];if(![0,2,5].includes(r))continue;const o=n.geom_bodyid[i],s=r===2;if(!this.bodies.has(o)){const d=new Ye;(s?this.robotRoot:this.terrainRoot).add(d),this.bodies.set(o,d)}const c=n.geom_matid[i]>=0?n.mat_rgba.subarray(n.geom_matid[i]*4,n.geom_matid[i]*4+4):n.geom_rgba.subarray(i*4,i*4+4),l=new da({color:s?new Ot(c[0],c[1],c[2]):new Ot("#293d43"),roughness:s?.48:.95,metalness:s?.28:0});this.materials.push(l);const h=new ze(bf(n,i,this.geometryCache),l);if(Er(h,n.geom_pos,n.geom_quat,i),h.castShadow=s,h.receiveShadow=!0,h.userData.geomId=i,this.bodies.get(o).add(h),s){if(!this.ghostBodies.has(o)){const f=new Ye;this.ghostRoot.add(f),this.ghostBodies.set(o,f)}const d=new ze(h.geometry,this.ghostMaterial);d.position.copy(h.position),d.quaternion.copy(h.quaternion),d.renderOrder=2,this.ghostBodies.get(o).add(d)}}this.target=new P,this.delta=new P,this.raycaster=new Sf,this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(e),this.resize(),this.setView("follow"),this.update()}resize(){const t=Math.max(1,this.container.clientWidth),e=Math.max(1,this.container.clientHeight);this.renderer.setSize(t,e),this.camera.aspect=t/e,this.camera.updateProjectionMatrix()}setView(t){if(!["follow","overview"].includes(t))throw new Error(`Unknown view ${t}`);this.view=t,t==="overview"?(this.camera.position.set(48,-64,63),this.controls.target.set(0,0,0)):(this.target.fromArray(this.physics.data.qpos,0),this.controls.target.copy(this.target),this.camera.position.copy(this.target).add(new P(2.4,-3.2,1.6))),this.controls.update()}update(){const t=this.physics.data;for(const[e,n]of this.bodies)Er(n,t.xpos,t.xquat,e);this.target.fromArray(t.qpos,0),this.sun.target.position.copy(this.target),this.sun.position.copy(this.target).add(new P(-6,-8,14)),this.view==="follow"&&(this.delta.copy(this.target).sub(this.controls.target).multiplyScalar(.12),this.controls.target.add(this.delta),this.camera.position.add(this.delta)),this.controls.update()}render(){this.update(),this.renderer.render(this.scene,this.camera)}showReferencePose(t){if(this.ghostRoot.visible=!!t,!t)return;const{physics:e}=this;this.ghostData||(this.ghostData=new e.mujoco.MjData(e.model));const n=this.ghostData;n.qpos.set(e.data.qpos),t.rootPosition&&n.qpos.set(t.rootPosition,0),t.rootQuaternionWxyz&&n.qpos.set(t.rootQuaternionWxyz,3),t.jointPosition&&e.joints.forEach((i,r)=>{n.qpos[i.qpos]=t.jointPosition[r]}),e.mujoco.mj_forward(e.model,n);for(const[i,r]of this.ghostBodies)Er(r,n.xpos,n.xquat,i)}pickGround(t,e){const n=this.renderer.domElement.getBoundingClientRect();this.raycaster.setFromCamera(new dt((t-n.left)/n.width*2-1,-(e-n.top)/n.height*2+1),this.camera);const i=this.raycaster.intersectObject(this.terrainRoot,!0)[0];return i?i.point.toArray():null}dispose(){var t;this.resizeObserver.disconnect(),this.controls.dispose(),(t=this.ghostData)==null||t.delete();for(const e of this.geometryCache.values())e.dispose();for(const e of this.materials)e.dispose();this.renderer.dispose(),this.renderer.domElement.remove()}}export{wf as PBFMViewer,wf as Viewer,bf as modelGeometry};
