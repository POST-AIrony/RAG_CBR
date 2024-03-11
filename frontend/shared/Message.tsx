import React from 'react';

interface MessageProps {
   text: string;
   isUserMessage: boolean;
}

const Message:React.FC<MessageProps> = ({text}) => {
    return (
        <div className="px-[15px] py-[10px] w-[55%] h-auto bg-[#ffffff] rounded-t-[10px] rounded-br-[10px] text-[#74777b] text-[1.375rem] font-medium shadow-md">
            <p>{text}</p>
        </div>
    );
};

export default Message;