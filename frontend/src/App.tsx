import React, {useState} from "react";
import Message from "../shared/Message.tsx";
function App() {
  const [requestText, setRequestText] = useState("");

  const handleNewRequest = () => {

  }
    
  return (
    <>
      <main className="relative grid grid-cols-10 w-full h-screen bg-[#0088bb]">
          <aside className="relative col-span-2 my-[50px] px-[20px] h-[calc(100%-100px)] border-r-2 border-r-[#ffffff]">
              <button className="absolute flex justify-center items-center bottom-0 ml-[calc(50%-45px)] w-[50px] h-[50px] bg-[#ffffff] rounded-[50%] shadow-md">
                  <img src="/icons/ClearChatIcon.svg" alt="" className="w-[25px] h-[25px]"/>
              </button>
          </aside>

          <section className="col-span-8 my-[50px] px-[50px]">
              <div className="w-full h-[calc(100vh-200px)]">
                  <Message text="No Яйца клац клац"/>
              </div>

              <div className="flex items-center w-full mt-[50px] h-[50px]">
                  <input type="text" placeholder="Введите запрос..." onChange={(event) => setRequestText(event.target.value)} className="px-[20px] w-[calc(100%-70px)] h-[50px] bg-transparent border-2 border-[#ffffff] rounded-[10px] text-[#ffffff] placeholder:text-[#ffffff] text-[1.375rem] font-medium outline-none"/>

                  <button onClick={handleNewRequest} className="flex justify-center items-center ml-[20px] w-[50px] h-[50px] bg-[#ffffff] rounded-[50%] shadow-md">
                      <img src="/icons/RightArrowIcon.svg" alt="" className="w-[25px] h-[25px]"/>
                  </button>
              </div>
          </section>
      </main>
    </>
  );
}

export default App;
